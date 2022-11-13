#pragma once

#include <dgl/aten/types.h>

#include "../graph/heterograph.h"

namespace dgl {
namespace inference {

std::pair<IdArray, IdArray> SortDstIds(int num_nodes,
                                       int num_devices_per_node,
                                       int batch_size,
                                       const IdArray& org_ids,
                                       const IdArray& part_ids,
                                       const IdArray& part_id_counts);

std::vector<IdArray> ExtractSrcIds(int num_nodes,
                                   int num_devices_per_node,
                                   int node_rank,
                                   int batch_size,
                                   const IdArray& org_ids,
                                   const IdArray& part_ids,
                                   const IdArray& part_id_counts);

} // namespace inference
} // namespace dgl
