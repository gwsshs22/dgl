import json
from pathlib import Path
import sys
import time

import torch

import dgl

from dgl.omega.omega_apis import partition_request
from dgl.omega.trace import trace_me

class SingleWorkerCommunicator:

    def __init__(self, worker_async_exec_context):
        self._req_id = 0
        self._async_exec_context = worker_async_exec_context

    def request(self, batch_id, batch_request):
        fut = self._async_exec_context.rpc_async().process(
            self._req_id,
            batch_id,
            batch_request.target_gnids,
            batch_request.target_features,
            batch_request.src_gnids,
            batch_request.dst_gnids
        )
        self._req_id += 1
        return fut

class WorkerGroupCommunicator:

    def __init__(
        self,
        num_machines,
        part_config_path,
        worker_async_exec_contexts):
        num_total_gpus_in_group = len(worker_async_exec_contexts)

        assert num_total_gpus_in_group % num_machines == 0
        self._num_machines = num_machines
        self._num_gpus_per_machine = num_total_gpus_in_group // num_machines
        self._async_exec_contexts = worker_async_exec_contexts
        self._req_id = 0

        self._load_part_config(part_config_path)

    def _load_part_config(self, part_config_path):
        self._part_config = json.loads(Path(part_config_path).read_text())
        assert len(self._part_config["node_map"]) == 1, "Support only homogeneous graphs currently."
        assert "_N" in self._part_config["node_map"], "Support only homogeneous graphs currently."
        nid_splits = self._part_config["node_map"]["_N"]
        nid_partitions = [s[0] for s in nid_splits]
        nid_partitions.append(nid_splits[-1][1])
        self._nid_partitions = torch.tensor(nid_partitions)

    def request(self, batch_id, batch_request):
        with trace_me(batch_id, "part_req"):
            partitioned_request = partition_request(
                self._num_machines,
                self._num_gpus_per_machine,
                self._nid_partitions,
                batch_request.target_gnids,
                batch_request.target_features,
                batch_request.src_gnids,
                batch_request.dst_gnids)

        target_gnids_list, target_features_list, src_gnids_list, dst_gnids_list = partitioned_request
        futs = []
        for i, async_exec_context in enumerate(self._async_exec_contexts):
            fut = async_exec_context.rpc_async().process(
                self._req_id,
                batch_id,
                target_gnids_list[i],
                target_features_list[i],
                src_gnids_list[i],
                dst_gnids_list[i])
            futs.append(fut)
        self._req_id += 1
        return torch.futures.collect_all(futs)

def create_worker_communicators(
    num_machines,
    num_gpus_per_machine,
    part_config_path,
    exec_mode,
    worker_async_exec_context_groups):

    ret_list = []
    for worker_async_exec_contexts in worker_async_exec_context_groups:
        assert len(worker_async_exec_contexts) % num_gpus_per_machine == 0
        if exec_mode == "dp":
            ret_list += [SingleWorkerCommunicator(w) for w in worker_async_exec_contexts]
        elif exec_mode == "cgp":
            ret_list += [WorkerGroupCommunicator(
                num_machines,
                part_config_path,
                worker_async_exec_contexts)]
        elif exec_mode == "cgp-multi":
            worker_comms = []
            for local_rank in range(num_gpus_per_machine):
                contexts = []
                itr = local_rank
                while itr < len(worker_async_exec_contexts):
                    contexts.append(worker_async_exec_contexts[itr])
                    itr += num_gpus_per_machine

                worker_comms.append(WorkerGroupCommunicator(
                    num_machines,
                    part_config_path,
                    contexts))

            ret_list += worker_comms
        else:
            raise f"Unknown exec_mode={exec_mode}"
            
    return ret_list


