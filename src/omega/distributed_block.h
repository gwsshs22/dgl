/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/distributed_block.h
 * \brief 
 */
#pragma once

#include <tuple>
#include <vector>

#include <dgl/base_heterograph.h>

namespace dgl {
namespace omega {

std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> ToDistributedBlocks(
    int num_machines, int machine_rank, int num_gpus_per_machine,
    const IdArray& target_gnids, const IdArray& src_gnids,
    const IdArray& src_part_ids, const IdArray& dst_gnids);

std::pair<HeteroGraphPtr, IdArray> ToDistributedBlock(
    int num_machines, int machine_rank, int num_gpus_per_machine, int local_gpu_idx,
    const IdArray& target_gnids, const IdArray& src_gnids,
    const IdArray& src_part_ids, const IdArray& dst_gnids);

}
}
