import sys

import torch
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

import dgl
from dgl.omega.omega_apis import (
    create_block,
    create_distributed_block,
)
from dgl.omega.dist_context import set_nccl_group, set_gloo_group

from models import create_model
from utils import init_torch_distributed

def gnn_executor_main(
    in_queue,
    out_queue,
    master_ip,
    master_dist_nccl_port,
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    num_omega_groups,
    omega_group_id,
    global_rank,
    local_rank,
    num_gpus_per_machine_in_group,
    gpu_ranks,
    local_gpu_rank_in_group,
    exec_mode,
    enable_comm_on_host,
    use_precoms,
    model_config,
    full_sampling,
    random_seed,
    tracing,
    device,
    wait_sem,
    finish_event
):

    if exec_mode != "dp":
        if enable_comm_on_host:
            gloo_group, gloo_group_ranks = init_torch_distributed(
                exec_mode,
                num_machines,
                num_gpus_per_machine,
                master_ip,
                master_dist_nccl_port,
                global_rank,
                local_rank,
                "gloo"
            )

            assert gloo_group_ranks == gpu_ranks
            set_gloo_group(gloo_group)
        else:
            nccl_group, nccl_group_ranks = init_torch_distributed(
                exec_mode,
                num_machines,
                num_gpus_per_machine,
                master_ip,
                master_dist_nccl_port,
                global_rank,
                local_rank,
                "nccl"
            )

            assert nccl_group_ranks == gpu_ranks
            set_nccl_group(nccl_group)

    exec_context = LocalExecutionContext(
        local_rank,
        exec_mode,
        use_precoms,
        model_config,
        random_seed)

    def run():
        with torch.no_grad():
            while True:

                def compute_one_req():
                    batch_id, target_features, block_data_list, global_in_degrees_list, src_inputs_list = in_queue.get()
                    if batch_id is None:
                        return None, None

                    blocks = []

                    for block_u, block_v, block_src_ids, block_num_srcs, block_num_dsts in block_data_list:
                        if exec_mode == "dp":
                            block = create_block(
                                block_u, block_v, block_src_ids, block_num_srcs, block_num_dsts
                            )
                        else:
                            block = create_distributed_block(
                                num_machines, machine_rank, num_gpus_per_machine_in_group, local_gpu_rank_in_group, gpu_ranks,
                                block_u, block_v, block_src_ids, block_num_srcs, block_num_dsts
                            )
                        blocks.append(block)

                    if exec_mode != "dp":
                        for i in range(len(blocks)):
                            blocks[i].set_in_degrees(global_in_degrees_list[i])

                    if full_sampling and use_precoms:
                        blocks *= model_config.num_layers

                    return batch_id, exec_context.execute(batch_id, target_features, blocks, src_inputs_list)

                batch_id, ret_tensor = compute_one_req()
                if batch_id is None:
                    break

                out_queue.put((batch_id, ret_tensor))
                wait_sem.acquire()

            out_queue.put((None, None))

    if tracing:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
        prof.export_chrome_trace(f"trace_gnn_{omega_group_id}_{global_rank}_{exec_mode}.json")
    else:
        run()

    finish_event.wait()


class LocalExecutionContext:

    def __init__(
        self,
        local_rank,
        exec_mode,
        use_precoms,
        model_config,
        random_seed):

        self._local_rank = local_rank
        self._exec_mode = exec_mode
        self._use_precoms = use_precoms
        self._model_config = model_config

        self._device = f"cuda:{self._local_rank}"

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self._load_model()

    def _load_model(self):
        gat_heads = [int(h) for h in self._model_config.gat_heads.split(",")]
        self._model = create_model(
            self._model_config.gnn,
            self._model_config.num_inputs,
            self._model_config.num_hiddens,
            self._model_config.num_classes,
            self._model_config.num_layers,
            gat_heads)

        self._model = self._model.to(self._device)

    def execute(self, batch_id, target_features, blocks, src_inputs_list):
        if self._use_precoms:
            return self.execute_with_precoms(batch_id, target_features, blocks, src_inputs_list)
        else:
            return self.execute_without_precoms(batch_id, target_features, blocks, src_inputs_list)
    
    def execute_without_precoms(self, batch_id, target_features, blocks, src_inputs_list):
        features = src_inputs_list[0]
        features = torch.concat((target_features, features))
        h = self._model(blocks, features)

        return h
    
    def execute_with_precoms(self, batch_id, target_features, blocks, src_inputs_list):
        num_layers = self._model_config.num_layers
        h = torch.concat((target_features, src_inputs_list[0]))
        for layer_idx in range(num_layers):
            h = self._model.layer_foward(layer_idx, blocks[layer_idx], h)
            if layer_idx != num_layers - 1:
                h = torch.concat((h, src_inputs_list[layer_idx + 1]))

        return h
