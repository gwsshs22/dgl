from .distributed_block import _set_nccl_group, _enable_comm_on_host

def set_nccl_group(nccl_group):
    _set_nccl_group(nccl_group)

def set_gloo_group(gloo_group):
    _enable_comm_on_host(gloo_group)
