/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/distributed_sampling.h
 * \brief 
 */
#pragma once

#include <tuple>
#include <vector>

#include <dgl/base_heterograph.h>

namespace dgl {
namespace omega {

std::pair<std::vector<IdArray>, std::vector<IdArray>> SplitLocalEdges(int num_machines,
                                                                      int machine_rank,
                                                                      int num_gpus_per_machine,
                                                                      int gpu_rank,
                                                                      const IdArray& global_src,
                                                                      const IdArray& global_dst,
                                                                      const IdArray& global_src_part_ids);

}
}
