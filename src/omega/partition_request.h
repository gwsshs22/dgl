/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/partition_request.h
 * \brief 
 */
#pragma once

#include <utility>
#include <vector>

#include <dgl/base_heterograph.h>

namespace dgl {
namespace omega {

std::pair<std::vector<IdArray>, std::vector<IdArray>> PartitionRequest(
    int num_machines,
    int num_gpus_per_machine,
    const IdArray& nid_partitions,
    const IdArray& num_assigned_targets_per_gpu,
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids);

}
}
