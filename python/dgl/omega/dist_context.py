from .distributed_block import _set_nccl_group

def set_nccl_group(nccl_group):
    _set_nccl_group(nccl_group)
