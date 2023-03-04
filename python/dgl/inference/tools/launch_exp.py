"""Launching tool for DGL distributed training"""
import os
import stat
import sys
import subprocess
import argparse
import signal
import logging
import time
import json
import multiprocessing
import queue
import re
import random
from functools import partial
from threading import Thread
from typing import Optional

def cleanup_proc(get_all_remote_pids, conn):
    '''This process tries to clean up the remote training tasks.
    '''
    print('cleanup process runs')
    # This process should not handle SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    data = conn.recv()
    # If the launch process exits normally, this process doesn't need to do anything.
    if data == 'exit':
        sys.exit(0)
    else:
        remote_pids = get_all_remote_pids()
        # Otherwise, we need to ssh to each machine and kill the training jobs.
        for (ip, port), pids in remote_pids.items():
            kill_process(ip, port, pids)
    print('cleanup process exits')

def kill_process(ip, port, pids):
    '''ssh to a remote machine and kill the specified processes.
    '''
    curr_pid = os.getpid()
    killed_pids = []
    # If we kill child processes first, the parent process may create more again. This happens
    # to Python's process pool. After sorting, we always kill parent processes first.
    pids.sort()
    for pid in pids:
        assert curr_pid != pid
        print('kill process {} on {}:{}'.format(pid, ip, port), flush=True)
        kill_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'kill {}\''.format(pid)
        subprocess.run(kill_cmd, shell=True)
        killed_pids.append(pid)
    # It's possible that some of the processes are not killed. Let's try again.
    for i in range(3):
        killed_pids = get_killed_pids(ip, port, killed_pids)
        if len(killed_pids) == 0:
            break
        else:
            killed_pids.sort()
            for pid in killed_pids:
                print('kill process {} on {}:{}'.format(pid, ip, port), flush=True)
                kill_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'kill -9 {}\''.format(pid)
                subprocess.run(kill_cmd, shell=True)

def get_killed_pids(ip, port, killed_pids):
    '''Get the process IDs that we want to kill but are still alive.
    '''
    killed_pids = [str(pid) for pid in killed_pids]
    killed_pids = ','.join(killed_pids)
    ps_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'ps -p {} -h\''.format(killed_pids)
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids = []
    for p in res.stdout.decode('utf-8').split('\n'):
        l = p.split()
        if len(l) > 0:
            pids.append(int(l[0]))
    return pids

def execute_remote(
    cmd: str,
    state_q: queue.Queue,
    ip: str,
    port: int,
    username: Optional[str] = ""
) -> Thread:
    """Execute command line on remote machine via ssh.

    Args:
        cmd: User-defined command (udf) to execute on the remote host.
        state_q: A queue collecting Thread exit states.
        ip: The ip-address of the host to run the command on.
        port: Port number that the host is listening on.
        thread_list:
        username: Optional. If given, this will specify a username to use when issuing commands over SSH.
            Useful when your infra requires you to explicitly specify a username to avoid permission issues.

    Returns:
        thread: The Thread whose run() is to run the `cmd` on the remote host. Returns when the cmd completes
            on the remote host.
    """
    ip_prefix = ""
    if username:
        ip_prefix += "{username}@".format(username=username)

    # Construct ssh command that executes `cmd` on the remote host
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -p {port} {ip_prefix}{ip} '{cmd}'".format(
        port=str(port),
        ip_prefix=ip_prefix,
        ip=ip,
        cmd=cmd,
    )

    # thread func to run the job
    def run(ssh_cmd, state_q):
        try:
            subprocess.check_call(ssh_cmd, shell=True)
            state_q.put(0)
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            state_q.put(err.returncode)
        except Exception:
            state_q.put(-1)

    thread = Thread(target=run, args=(ssh_cmd, state_q,))
    thread.setDaemon(True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread

def get_remote_pids(ip, port, cmd_regex):
    """Get the process IDs that run the command in the remote machine.
    """
    pids = []
    curr_pid = os.getpid()
    # Here we want to get the python processes. We may get some ssh processes, so we should filter them out.
    ps_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'ps -aux | grep python | grep -v StrictHostKeyChecking\''
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    for p in res.stdout.decode('utf-8').split('\n'):
        l = p.split()
        if len(l) < 2:
            continue
        # We only get the processes that run the specified command.
        res = re.search(cmd_regex, p)
        if res is not None and int(l[1]) != curr_pid:
            pids.append(l[1])

    pid_str = ','.join([str(pid) for pid in pids])
    ps_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'pgrep -P {}\''.format(pid_str)
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids1 = res.stdout.decode('utf-8').split('\n')
    all_pids = []
    for pid in set(pids + pids1):
        if pid == '' or int(pid) == curr_pid:
            continue
        all_pids.append(int(pid))
    all_pids.sort()
    return all_pids

def get_all_remote_pids(hosts, ssh_port, cmd_regex):
    '''Get all remote processes.
    '''
    remote_pids = {}
    for node_id, host in enumerate(hosts):
        ip, _ = host
        pids = get_remote_pids(ip, ssh_port, cmd_regex)
        remote_pids[(ip, ssh_port)] = pids
    return remote_pids

def get_available_port(ip):
    """Get available port with specified ip."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    num_trials = 1000
    for _ in range(num_trials):
        port = random.randint(30000, 65535)
        try:
            sock.connect((ip, port))
        except:
            return port
    raise RuntimeError("Failed to get available port for ip~{}".format(ip))

def make_master_process_cmd(args, master_host, master_port, master_torch_port, num_nodes):
    num_backup_servers = args.num_backup_servers if args.num_backup_servers else num_nodes * args.num_samplers_per_node
    if args.parallelization_type == "vcut":
        num_backup_servers = 0
    python_exec = args.python_exec
    iface = args.iface
    if os.environ.get("PYTHONPATH", ""):
        cmd = f"PYTHONPATH={os.environ.get('PYTHONPATH')} "
    else:
        cmd = ""
    cmd = cmd + "NCCL_IB_DISABLE=1 " # Fixme
    cmd = cmd + f"""NCCL_SOCKET_IFNAME={iface} GLOO_SOCKET_IFNAME={iface} {python_exec} -m dgl.inference.main \
--role master \
--master-host {master_host} \
--master-port {master_port} \
--master-torch-port {master_torch_port} \
--input-trace-dir {args.input_trace_dir} \
--num-warmups {args.num_warmups} \
--num-requests {args.num_requests} \
--result-dir {args.result_dir} \
--exp-lambda {args.exp_lambda} \
--node-rank 0 \
--num-nodes {num_nodes} \
--num-backup-servers {num_backup_servers} \
--num-devices-per-node {args.num_devices_per_node} \
--num-samplers-per-node {args.num_samplers_per_node} \
--ip-config-path {args.ip_config} \
--graph-name {args.graph_name} \
--graph-config-path {args.part_config} \
--iface {iface} \
--parallelization-type {args.parallelization_type} \
--model {args.model} \
--num-layers {args.num_layers} \
--fanouts "{args.fanouts}" \
--num-heads "{args.num_heads}" \
--num-inputs {args.num_inputs} \
--num-hiddens {args.num_hiddens} \
--num-outputs {args.num_outputs}"""
    if args.using_precomputed_aggregations:
        cmd += " --using-precomputed-aggregations"
        cmd += f" --precom-filename {args.precom_filename}"
    if args.collect_stats:
        cmd += " --collect-stats"
    if args.execute_one_by_one:
        cmd += " --execute-one-by-one"

    return cmd

def make_worker_process_cmd(args, master_host, master_port, master_torch_port, num_nodes, worker_idx):
    num_backup_servers = args.num_backup_servers if args.num_backup_servers else num_nodes * args.num_samplers_per_node
    if args.parallelization_type == "vcut":
        num_backup_servers = 0    
    python_exec = args.python_exec
    iface = args.iface
    if os.environ.get("PYTHONPATH", ""):
        cmd = f"PYTHONPATH={os.environ.get('PYTHONPATH')} "
    else:
        cmd = ""
    cmd = cmd + "NCCL_IB_DISABLE=1 " # Fixme: this should be given as an argument
    cmd = cmd + f"""NCCL_SOCKET_IFNAME={iface} GLOO_SOCKET_IFNAME={iface} {python_exec} -m dgl.inference.main \
--role worker \
--master-host {master_host} \
--master-port {master_port} \
--master-torch-port {master_torch_port} \
--result-dir {args.result_dir} \
--exp-lambda {args.exp_lambda} \
--node-rank {worker_idx} \
--num-nodes {num_nodes} \
--num-backup-servers {num_backup_servers} \
--num-devices-per-node {args.num_devices_per_node} \
--num-samplers-per-node {args.num_samplers_per_node} \
--ip-config-path {args.ip_config} \
--graph-name {args.graph_name} \
--graph-config-path {args.part_config} \
--iface {iface} \
--parallelization-type {args.parallelization_type} \
--model {args.model} \
--num-layers {args.num_layers} \
--fanouts "{args.fanouts}" \
--num-heads "{args.num_heads}" \
--num-inputs {args.num_inputs} \
--num-hiddens {args.num_hiddens} \
--num-outputs {args.num_outputs}"""
    if args.using_precomputed_aggregations:
        cmd += " --using-precomputed-aggregations"
        cmd += f" --precom-filename {args.precom_filename}"
    if args.collect_stats:
        cmd += " --collect-stats"
    if args.execute_one_by_one:
        cmd += " --execute-one-by-one"

    return cmd

def submit_jobs(args, dry_run=False):
    """Submit distributed jobs (server and client processes) via ssh"""
    if dry_run:
        print("Currently it's in dry run mode which means no jobs will be launched.")
    servers_cmd = []
    clients_cmd = []
    hosts = []
    thread_list = []
    server_count_per_machine = 0

    # Get the IP addresses of the cluster.
    ip_config = args.ip_config
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) == 2:
                ip = result[0]
                port = int(result[1])
                hosts.append((ip, port))
            elif len(result) == 1:
                ip = result[0]
                port = get_available_port(ip)
                hosts.append((ip, port))
            else:
                raise RuntimeError("Format error of ip_config.")
    # Get partition info of the graph data
    part_config = args.part_config
    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert 'num_parts' in part_metadata, 'num_parts does not exist.'
    # The number of partitions must match the number of machines in the cluster.
    assert part_metadata['num_parts'] == len(hosts), \
            'The number of graph partitions has to match the number of machines in the cluster.'

    master_host = hosts[0][0]
    master_port = get_available_port(master_host)
    master_torch_port = get_available_port(master_host)
    num_nodes = len(hosts)
    state_q = queue.Queue()

    # Launch master process
    cmd = make_master_process_cmd(args, master_host, master_port, master_torch_port, num_nodes)
    if not dry_run:
        thread_list.append(execute_remote(cmd, state_q, master_host, args.ssh_port, username=args.ssh_username))
    else:
        print(f"master_process_command={cmd}")
    
    for i in range(1, len(hosts)):
        cmd = make_worker_process_cmd(args, master_host, master_port, master_torch_port, num_nodes, i)
        if not dry_run:
            thread_list.append(execute_remote(cmd, state_q, hosts[i][0], args.ssh_port, username=args.ssh_username))
        else:
            print(f"worker_process_{i}_command={cmd}")


    # Start a cleanup process dedicated for cleaning up remote training jobs.
    conn1,conn2 = multiprocessing.Pipe()
    func = partial(get_all_remote_pids, hosts, args.ssh_port, ".*python.*-m.*dgl.inference.main*|.*python.*-m.*dgl.inference.fork*")
    process = multiprocessing.Process(target=cleanup_proc, args=(func, conn1))
    process.start()

    def signal_handler(signal, frame):
        logging.info('Stop launcher')
        # We need to tell the cleanup process to kill remote training jobs.
        conn2.send('cleanup')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    err = 0
    for thread in thread_list:
        thread.join()
        err_code = state_q.get()
        if err_code != 0:
            # Record err_code
            # We record one of the error if there are multiple
            err = err_code

    # The training processes complete. We should tell the cleanup process to exit.
    conn2.send('exit')
    process.join()
    if err != 0:
        print("Task failed")
        sys.exit(-1)

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--ssh_port', type=int, default=22, help='SSH Port.')
    parser.add_argument(
        "--ssh_username", default="",
        help="Optional. When issuing commands (via ssh) to cluster, use the provided username in the ssh cmd. "
             "Example: If you provide --ssh_username=bob, then the ssh command will be like: 'ssh bob@1.2.3.4 CMD' "
             "instead of 'ssh 1.2.3.4 CMD'"
    )
    parser.add_argument('--dry_run', action="store_true")
    parser.add_argument('--python_exec', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_devices_per_node', type=int, required=True)
    parser.add_argument('--num_samplers_per_node', type=int, required=True)
    parser.add_argument('--ip_config', type=str, required=True)
    parser.add_argument('--num_backup_servers', type=int)
    parser.add_argument('--part_config', type=str, required=True)
    parser.add_argument('--graph_name', help="reddit, ogbn-papers100M, ...", required=True)
    parser.add_argument('--iface', type=str, required=True)
    parser.add_argument('--parallelization_type', type=str, choices=["data", "p3", "vcut"], required=True)
    parser.add_argument('--model', type=str, choices=['gcn', 'sage', 'gat'], required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--fanouts', type=str, default=" -1,-1", required=False)
    parser.add_argument('--num_inputs', type=int, required=True)
    parser.add_argument('--num_heads', type=str, default=" 8,8", required=False)
    parser.add_argument('--num_hiddens', type=int, required=True)
    parser.add_argument('--num_outputs', type=int, required=True)
    parser.add_argument('--using_precomputed_aggregations', action='store_true')
    parser.add_argument('--precom_filename', type=str)

    parser.add_argument('--input_trace_dir', type=str, required=True)
    parser.add_argument('--num_warmups', type=int, required=True)
    parser.add_argument('--num_requests', type=int, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--collect_stats', action='store_true')
    parser.add_argument('--execute_one_by_one', action='store_true')
    parser.add_argument('--exp_lambda', type=float, default=0.0)

    args = parser.parse_args()
    print(args)
    submit_jobs(args, dry_run=args.dry_run)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()
