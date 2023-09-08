"""Launching tool for DGL distributed training"""
import argparse
import json
import logging
import multiprocessing
import os
import queue
import random
import re
import signal
import stat
import subprocess
import sys
import time
from functools import partial
from threading import Thread
from typing import Optional


def cleanup_proc(get_all_remote_pids, conn):
    """This process tries to clean up the remote training tasks."""
    print("cleanupu process runs")
    # This process should not handle SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    data = conn.recv()
    # If the launch process exits normally, this process doesn't need to do anything.
    if data == "exit":
        sys.exit(0)
    else:
        remote_pids = get_all_remote_pids()
        # Otherwise, we need to ssh to each machine and kill the training jobs.
        for (ip, port), pids in remote_pids.items():
            kill_process(ip, port, pids)
    print("cleanup process exits")


def kill_process(ip, port, pids):
    """ssh to a remote machine and kill the specified processes."""
    curr_pid = os.getpid()
    killed_pids = []
    # If we kill child processes first, the parent process may create more again. This happens
    # to Python's process pool. After sorting, we always kill parent processes first.
    pids.sort()
    for pid in pids:
        assert curr_pid != pid
        print("kill process {} on {}:{}".format(pid, ip, port), flush=True)
        kill_cmd = (
            "ssh -o StrictHostKeyChecking=no -p "
            + str(port)
            + " "
            + ip
            + " 'kill {}'".format(pid)
        )
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
                print(
                    "kill process {} on {}:{}".format(pid, ip, port), flush=True
                )
                kill_cmd = (
                    "ssh -o StrictHostKeyChecking=no -p "
                    + str(port)
                    + " "
                    + ip
                    + " 'kill -9 {}'".format(pid)
                )
                subprocess.run(kill_cmd, shell=True)


def get_killed_pids(ip, port, killed_pids):
    """Get the process IDs that we want to kill but are still alive."""
    killed_pids = [str(pid) for pid in killed_pids]
    killed_pids = ",".join(killed_pids)
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'ps -p {} -h'".format(killed_pids)
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids = []
    for p in res.stdout.decode("utf-8").split("\n"):
        l = p.split()
        if len(l) > 0:
            pids.append(int(l[0]))
    return pids


def execute_remote(
    cmd: str,
    state_q: queue.Queue,
    ip: str,
    port: int,
    username: Optional[str] = "",
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

    thread = Thread(
        target=run,
        args=(
            ssh_cmd,
            state_q,
        ),
    )
    thread.setDaemon(True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread


def get_remote_pids(ip, port, cmd_regex):
    """Get the process IDs that run the command in the remote machine."""
    pids = []
    curr_pid = os.getpid()
    # Here we want to get the python processes. We may get some ssh processes, so we should filter them out.
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'ps -aux | grep python | grep -v StrictHostKeyChecking'"
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    for p in res.stdout.decode("utf-8").split("\n"):
        l = p.split()
        if len(l) < 2:
            continue
        # We only get the processes that run the specified command.
        res = re.search(cmd_regex, p)
        if res is not None and int(l[1]) != curr_pid:
            pids.append(l[1])

    pid_str = ",".join([str(pid) for pid in pids])
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'pgrep -P {}'".format(pid_str)
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids1 = res.stdout.decode("utf-8").split("\n")
    all_pids = []
    for pid in set(pids + pids1):
        if pid == "" or int(pid) == curr_pid:
            continue
        all_pids.append(int(pid))
    all_pids.sort()
    return all_pids


def get_all_remote_pids(hosts, ssh_port, command_regex):
    """Get all remote processes."""
    remote_pids = {}
    for node_id, host in enumerate(hosts):
        ip, _ = host
        # When creating training processes in remote machines, we may insert some arguments
        # in the commands. We need to use regular expressions to match the modified command.
        pids = get_remote_pids(ip, ssh_port, command_regex)
        remote_pids[(ip, ssh_port)] = pids
    return remote_pids



def construct_dgl_server_env_vars(
    num_samplers: int,
    server_omp_threads: int,
    tot_num_clients: int,
    part_config: str,
    ip_config: str,
    num_servers: int,
    graph_format: str,
    keep_alive: bool,
    pythonpath: Optional[str] = "",
) -> str:
    """Constructs the DGL server-specific env vars string that are required for DGL code to behave in the correct
    server role.
    Convenience function.

    Args:
        num_samplers:
        server_omp_threads:
        tot_num_clients:
        part_config: Partition config.
            Relative path to workspace.
        ip_config: IP config file containing IP addresses of cluster hosts.
            Relative path to workspace.
        num_servers:
        graph_format:
        keep_alive:
            Whether to keep server alive when clients exit
        pythonpath: Optional. If given, this will pass this as PYTHONPATH.

    Returns:
        server_env_vars: The server-specific env-vars in a string format, friendly for CLI execution.

    """
    server_env_vars_template = (
        "DGL_ROLE={DGL_ROLE} "
        "DGL_NUM_SAMPLER={DGL_NUM_SAMPLER} "
        "OMP_NUM_THREADS={OMP_NUM_THREADS} "
        "DGL_NUM_CLIENT={DGL_NUM_CLIENT} "
        "DGL_CONF_PATH={DGL_CONF_PATH} "
        "DGL_IP_CONFIG={DGL_IP_CONFIG} "
        "DGL_NUM_SERVER={DGL_NUM_SERVER} "
        "DGL_GRAPH_FORMAT={DGL_GRAPH_FORMAT} "
        "DGL_KEEP_ALIVE={DGL_KEEP_ALIVE} "
        "{suffix_optional_envvars}"
    )
    suffix_optional_envvars = ""
    if pythonpath:
        suffix_optional_envvars += f"PYTHONPATH={pythonpath} "
    return server_env_vars_template.format(
        DGL_ROLE="server",
        DGL_NUM_SAMPLER=num_samplers,
        OMP_NUM_THREADS=server_omp_threads,
        DGL_NUM_CLIENT=tot_num_clients,
        DGL_CONF_PATH=part_config,
        DGL_IP_CONFIG=ip_config,
        DGL_NUM_SERVER=num_servers,
        DGL_GRAPH_FORMAT=graph_format,
        DGL_KEEP_ALIVE=int(keep_alive),
        suffix_optional_envvars=suffix_optional_envvars,
    )


def construct_dgl_client_env_vars(
    num_samplers: int,
    tot_num_clients: int,
    part_config: str,
    ip_config: str,
    num_servers: int,
    graph_format: str,
    group_id: int,
    pythonpath: Optional[str] = "",
) -> str:
    """Constructs the DGL client-specific env vars string that are required for DGL code to behave in the correct
    client role.
    Convenience function.

    Args:
        num_samplers:
        tot_num_clients:
        part_config: Partition config.
            Relative path to workspace.
        ip_config: IP config file containing IP addresses of cluster hosts.
            Relative path to workspace.
        num_servers:
        graph_format:
        group_id:
            Used in client processes to indicate which group it belongs to.
        pythonpath: Optional. If given, this will pass this as PYTHONPATH.

    Returns:
        client_env_vars: The client-specific env-vars in a string format, friendly for CLI execution.

    """
    client_env_vars_template = (
        "DGL_DIST_MODE={DGL_DIST_MODE} "
        "DGL_ROLE={DGL_ROLE} "
        "DGL_NUM_SAMPLER={DGL_NUM_SAMPLER} "
        "DGL_NUM_CLIENT={DGL_NUM_CLIENT} "
        "DGL_CONF_PATH={DGL_CONF_PATH} "
        "DGL_IP_CONFIG={DGL_IP_CONFIG} "
        "DGL_NUM_SERVER={DGL_NUM_SERVER} "
        "DGL_GRAPH_FORMAT={DGL_GRAPH_FORMAT} "
        "DGL_GROUP_ID={DGL_GROUP_ID} "
        "{suffix_optional_envvars}"
    )
    # append optional additional env-vars
    suffix_optional_envvars = ""
    if pythonpath:
        suffix_optional_envvars += f"PYTHONPATH={pythonpath} "
    return client_env_vars_template.format(
        DGL_DIST_MODE="distributed",
        DGL_ROLE="client",
        DGL_NUM_SAMPLER=num_samplers,
        DGL_NUM_CLIENT=tot_num_clients,
        DGL_CONF_PATH=part_config,
        DGL_IP_CONFIG=ip_config,
        DGL_NUM_SERVER=num_servers,
        DGL_GRAPH_FORMAT=graph_format,
        DGL_GROUP_ID=group_id,
        suffix_optional_envvars=suffix_optional_envvars,
    )


def wrap_cmd_with_local_envvars(cmd: str, env_vars: str) -> str:
    """Wraps a CLI command with desired env vars with the following properties:
    (1) env vars persist for the entire `cmd`, even if it consists of multiple "chained" commands like:
        cmd = "ls && pwd && python run/something.py"
    (2) env vars don't pollute the environment after `cmd` completes.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = "VAR1=value1 VAR2=value2"
        >>> wrap_cmd_with_local_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Args:
        cmd:
        env_vars: A string containing env vars, eg "VAR1=val1 VAR2=val2"

    Returns:
        cmd_with_env_vars:

    """
    # use `export` to persist env vars for entire cmd block. required if udf_command is a chain of commands
    # also: wrap in parens to not pollute env:
    #     https://stackoverflow.com/a/45993803
    return f"(export {env_vars}; {cmd})"


def wrap_cmd_with_extra_envvars(cmd: str, env_vars: list) -> str:
    """Wraps a CLI command with extra env vars

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = ["VAR1=value1", "VAR2=value2"]
        >>> wrap_cmd_with_extra_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Args:
        cmd:
        env_vars: A list of strings containing env vars, e.g., ["VAR1=value1", "VAR2=value2"]

    Returns:
        cmd_with_env_vars:
    """
    env_vars = " ".join(env_vars)
    return wrap_cmd_with_local_envvars(cmd, env_vars)

def get_available_port(ip, num_required_ports=1):
    """Get available port with specified ip."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ports = []

    for _ in range(num_required_ports):
        for _ in range(10):
            port = random.randint(12345, 65535)
            if port in ports:
                continue
            try:
                sock.connect((ip, port))
            except:
                ports.append(port)
                break
    
    if len(ports) != num_required_ports:
        raise RuntimeError("Failed to get available port for ip~{}".format(ip))

    if num_required_ports == 1:
        return ports[0]
    else:
        return ports

def submit_jobs(args, dry_run=False):
    """Submit distributed jobs (server and client processes) via ssh"""
    if dry_run:
        print(
            "Currently it's in dry run mode which means no jobs will be launched."
        )
    num_servers = 1
    servers_cmd = []
    master_cmd = []
    workers_cmd = []

    hosts = []
    thread_list = []
    server_count_per_machine = 0

    # Get the IP addresses of the cluster.
    ip_config = os.path.join(args.workspace, args.ip_config)
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
            server_count_per_machine = num_servers
    # Get partition info of the graph data
    part_config = os.path.join(args.workspace, args.part_config)
    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert "num_parts" in part_metadata, "num_parts does not exist."
    # The number of partitions must match the number of machines in the cluster.
    assert part_metadata["num_parts"] == len(
        hosts
    ), "The number of graph partitions has to match the number of machines in the cluster."

    graph_name = part_metadata["graph_name"]
    num_samplers = args.num_omega_groups - 1
    state_q = queue.Queue()
    tot_num_clients = args.num_gpus_per_machine * (1 + num_samplers) * len(hosts)

    master_addr = hosts[0][0]
    master_ports = get_available_port(master_addr, 1 + args.num_omega_groups) 
    master_rpc_port = master_ports[0]
    master_dist_comm_ports = ",".join(map(lambda p: str(p), master_ports[1:]))

    
    server_env_vars = construct_dgl_server_env_vars(
        num_samplers=num_samplers,
        server_omp_threads=args.server_omp_threads,
        tot_num_clients=tot_num_clients,
        part_config=part_config,
        ip_config=ip_config,
        num_servers=num_servers,
        graph_format=args.graph_format,
        keep_alive=args.keep_alive,
        pythonpath=os.environ.get("PYTHONPATH", ""),
    )
    for i in range(len(hosts) * server_count_per_machine):
        ip, _ = hosts[int(i / server_count_per_machine)]
        server_env_vars_cur = f"{server_env_vars} DGL_SERVER_ID={i}"
        server_command = (
            f"{args.python_bin} {args.dgl_home}/omega/serving/graph_server.py " +
            f"--ip_config {ip_config} " +
            f"--num_layers {args.num_layers} " +
            f"--num_hiddens {args.num_hiddens} " +
            f"--exec_mode {args.exec_mode} " +
            (f"--feature_dim {args.feature_dim} " if args.feature_dim else "") +
            (f"--use_precoms  " if args.use_precoms else "") +
            (f"--precom_path {args.precom_path} " if args.precom_path else "") +
            f"--random_seed {args.random_seed} "
            f"--master_ip {master_addr} " +
            f"--master_rpc_port {master_rpc_port} " +
            f"--num_omega_groups {args.num_omega_groups} " +
            f"--num_machines {len(hosts)} " +
            f"--num_gpus_per_machine {args.num_gpus_per_machine} " +
            f"--machine_rank {i} "
        )

        cmd = wrap_cmd_with_local_envvars(server_command, server_env_vars_cur)
        cmd = (
            wrap_cmd_with_extra_envvars(cmd, args.extra_envs)
            if len(args.extra_envs) > 0
            else cmd
        )
        servers_cmd.append(cmd)
        if not dry_run:
            thread_list.append(
                execute_remote(
                    cmd,
                    state_q,
                    ip,
                    args.ssh_port,
                    username=args.ssh_username,
                )
            )

    # launch client tasks
    client_env_vars = construct_dgl_client_env_vars(
        num_samplers=num_samplers,
        tot_num_clients=tot_num_clients,
        part_config=part_config,
        ip_config=ip_config,
        num_servers=num_servers,
        graph_format=args.graph_format,
        group_id=0,
        pythonpath=os.environ.get("PYTHONPATH", ""),
    )

    # Launch master
    master_command = (
        f"{args.python_bin} {args.dgl_home}/omega/serving/master.py " +
        f"--ip_config {ip_config} " +
        f"--master_ip {master_addr} " +
        f"--master_rpc_port {master_rpc_port} " +
        f"--master_dist_comm_ports {master_dist_comm_ports} " +
        f"--num_omega_groups {args.num_omega_groups} " +
        f"--num_machines {len(hosts)} " +
        f"--num_gpus_per_machine {args.num_gpus_per_machine} " +
        f"--graph_name {graph_name} "
        f"--part_config_path {part_config} " +
        f"--worker_num_sampler_threads {args.worker_num_sampler_threads} " +
        f"--exec_mode {args.exec_mode} " +
        (f"--feature_dim {args.feature_dim} " if args.feature_dim else "") +
        (f"--use_precoms " if args.use_precoms else "") +
        f"--trace_dir {args.trace_dir} " +
        (f"--profiling " if args.profiling else "") +
        f"--exp_type {args.exp_type} " +
        (f"--num_reqs {args.num_reqs} " if args.num_reqs else "") +
        (f"--req_per_sec {args.req_per_sec} " if args.req_per_sec else "") +
        (f"--exp_secs {args.exp_secs} " if args.exp_secs else "") +
        (f"--result_dir {args.result_dir} " if args.result_dir else "") +
        (f"--tracing " if args.tracing else "") +
        f"--arrival_type {args.arrival_type} " +
        f"--gnn {args.gnn} " +
        f"--num_inputs {args.num_inputs} " +
        f"--num_hiddens {args.num_hiddens} " +
        f"--num_classes {args.num_classes} " +
        f"--num_layers {args.num_layers} " +
        f"--gat_heads {args.gat_heads} " +
        f"--fanouts {args.fanouts} " +
        f"--random_seed {args.random_seed} "
    )

    cmd = (
        wrap_cmd_with_extra_envvars(master_command, args.extra_envs + [f"OMP_NUM_THREADS={args.master_omp_threads}"])
        if len(args.extra_envs) > 0
        else master_command
    )
    master_cmd.append(cmd)
    if not dry_run:
        thread_list.append(
            execute_remote(
                cmd,
                state_q,
                master_addr,
                args.ssh_port,
                username=args.ssh_username,
            )
        )

    for omega_group_id in range(args.num_omega_groups):
        for node_id, host in enumerate(hosts):
            ip, _ = host
            for local_rank in range(args.num_gpus_per_machine):
                worker_command = (
                    f"{args.python_bin} {args.dgl_home}/omega/serving/worker.py "
                    f"--master_ip {master_addr} "
                    f"--master_rpc_port {master_rpc_port} "
                    f"--num_omega_groups {args.num_omega_groups} "
                    f"--omega_group_id {omega_group_id} "
                    f"--num_machines {len(hosts)} "
                    f"--machine_rank {node_id} "
                    f"--num_gpus_per_machine {args.num_gpus_per_machine} "
                    f"--local_rank {local_rank} "
                )

                cmd = wrap_cmd_with_local_envvars(
                    worker_command, client_env_vars
                )
                cmd = (
                    wrap_cmd_with_extra_envvars(cmd, args.extra_envs + [f"OMP_NUM_THREADS={args.worker_omp_threads}"])
                    if len(args.extra_envs) > 0
                    else cmd
                )
                workers_cmd.append(cmd)
                if not dry_run:
                    thread_list.append(
                        execute_remote(
                            cmd, state_q, ip, args.ssh_port, username=args.ssh_username
                        )
                    )

    # return commands of clients/servers directly if in dry run mode
    if dry_run:
        return servers_cmd, master_cmd, workers_cmd

    # Start a cleanup process dedicated for cleaning up remote training jobs.
    conn1, conn2 = multiprocessing.Pipe()
    func = partial(get_all_remote_pids, hosts, args.ssh_port, f"{args.python_bin}.*{args.dgl_home}/omega/serving.*")
    process = multiprocessing.Process(target=cleanup_proc, args=(func, conn1))
    process.start()

    def signal_handler(signal, frame):
        logging.info("Stop launcher")
        # We need to tell the cleanup process to kill remote training jobs.
        conn2.send("cleanup")
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
    conn2.send("exit")
    process.join()
    if err != 0:
        print("Task failed")
        sys.exit(-1)

def main():
    parser = argparse.ArgumentParser(description="Launch a distributed job")
    parser.add_argument("--ssh_port", type=int, default=22, help="SSH Port.")
    parser.add_argument(
        "--ssh_username",
        default="",
        help="Optional. When issuing commands (via ssh) to cluster, use the provided username in the ssh cmd. "
        "Example: If you provide --ssh_username=bob, then the ssh command will be like: 'ssh bob@1.2.3.4 CMD' "
        "instead of 'ssh 1.2.3.4 CMD'",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Path of user directory of distributed tasks. \
                        This is used to specify a destination location where \
                        the contents of current directory will be rsyncd",
    )
    parser.add_argument('--feature_dim', type=int)
    parser.add_argument("--master_omp_threads", type=int, default=8)
    parser.add_argument("--worker_num_sampler_threads", type=int, default=16)
    parser.add_argument("--worker_omp_threads", type=int, default=8)
    parser.add_argument('--exec_mode', type=str, choices=["dp", "cgp", "cgp-multi"])
    parser.add_argument('--trace_dir', type=str, required=True)
    parser.add_argument('--exp_type', type=str, choices=["latency", "throughput"], required=True)

    parser.add_argument('--profiling', action="store_true")
    parser.add_argument('--tracing', action="store_true")
    parser.add_argument('--result_dir', type=str)
    # For latency exp
    parser.add_argument('--num_reqs', type=int)
    # For throughput exp
    parser.add_argument('--req_per_sec', type=float)
    parser.add_argument('--exp_secs', type=float)
    parser.add_argument('--arrival_type', choices=['poisson', 'uniform'], default='poisson')
    parser.add_argument("--use_precoms", action="store_true")

    # Model configuration
    parser.add_argument('--gnn', type=str, required=True)
    parser.add_argument('--num_inputs', type=int, required=True)
    parser.add_argument('--num_hiddens', type=int, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--gat_heads', type=str, required=True)
    parser.add_argument('--fanouts', type=str, required=True)

    parser.add_argument('--random_seed', type=int, default=5123412)

    parser.add_argument("--dgl_home", type=str, required=True)
    parser.add_argument("--python_bin", type=str, required=True)
    parser.add_argument("--num_omega_groups", type=int, default=1)
    parser.add_argument(
        "--num_gpus_per_machine",
        type=int,
    )

    parser.add_argument(
        "--part_config",
        type=str,
        help="The file (in workspace) of the partition config",
    )
    parser.add_argument(
        "--ip_config",
        type=str,
        help="The file (in workspace) of IP configuration for server processes",
    )
    parser.add_argument(
        "--server_omp_threads",
        type=int,
        default=16,
        help="The number of OMP threads in the server process. \
                        It should be small if server processes and trainer processes run on \
                        the same machine. By default, it is 16.",
    )
    parser.add_argument(
        "--graph_format",
        type=str,
        default="csc",
        help='The format of the graph structure of each partition. \
                        The allowed formats are csr, csc and coo. A user can specify multiple \
                        formats, separated by ",". For example, the graph format is "csr,csc".',
    )
    parser.add_argument(
        "--extra_envs",
        nargs="+",
        type=str,
        default=[],
        help="Extra environment parameters need to be set. For example, \
                        you can set the LD_LIBRARY_PATH and NCCL_DEBUG by adding: \
                        --extra_envs LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH NCCL_DEBUG=INFO ",
    )
    parser.add_argument(
        "--keep_alive",
        action="store_true",
        help="Servers keep alive when clients exit",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        help="Used to check whether there exist alive servers",
    )
    parser.add_argument("--precom_path", type=str, default="")
    args = parser.parse_args()
    if args.keep_alive:
        assert (
            args.server_name is not None
        ), "Server name is required if '--keep_alive' is enabled."
        print("Servers will keep alive even clients exit...")
    assert (
        args.num_gpus_per_machine is not None and args.num_gpus_per_machine > 0
    ), "--num_gpus_per_machine must be a positive number."
    assert (
        args.server_omp_threads > 0
    ), "--server_omp_threads must be a positive number."
    assert (
        args.workspace is not None
    ), "A user has to specify a workspace with --workspace."
    assert (
        args.part_config is not None
    ), "A user has to specify a partition configuration file with --part_config."
    assert (
        args.ip_config is not None
    ), "A user has to specify an IP configuration file with --ip_config."

    if args.exec_mode == "cgp" or args.exec_mode == "cgp-multi":
        assert args.use_precoms
    
    if args.tracing:
        assert args.result_dir is not None

    submit_jobs(args)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()
