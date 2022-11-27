#pragma once

#include <dgl/aten/types.h>

#include "../graph/heterograph.h"

namespace dgl {
namespace inference {

std::pair<std::vector<IdArray>, std::vector<IdArray>> SplitLocalEdges(int num_nodes,
                                                                     const IdArray& global_src,
                                                                     const IdArray& global_dst,
                                                                     const IdArray& global_src_part_ids);


std::tuple<IdArray, IdArray, IdArray> SortDstIds(int num_nodes,
                                                 int num_devices_per_node,
                                                 int batch_size,
                                                 const IdArray& org_ids,
                                                 const IdArray& part_ids,
                                                 const IdArray& part_id_counts);

std::pair<std::vector<IdArray>, std::vector<IdArray>> ExtractSrcIds(int num_nodes,
                                                                    int num_devices_per_node,
                                                                    int node_rank,
                                                                    int batch_size,
                                                                    const IdArray& org_ids,
                                                                    const IdArray& part_ids,
                                                                    const IdArray& part_id_counts);


std::vector<HeteroGraphPtr> SplitBlocks(const HeteroGraphRef& graph_ref,
                                        int num_devices_per_node,
                                        const std::vector<IdArray>& sorted_src_bids_list,
                                        const IdArray& sorted_dst_bids);

} // namespace inference
} // namespace dgl
