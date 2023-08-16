#include "distributed_sampling.h"

#include <dgl/runtime/parallel_for.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace omega {

std::pair<std::vector<IdArray>, std::vector<IdArray>> SplitLocalEdges(int num_machines,
                                                                      int machine_rank,
                                                                      int num_gpus_per_machine,
                                                                      int gpu_rank,
                                                                      const IdArray& global_src,
                                                                      const IdArray& global_dst,
                                                                      const IdArray& global_src_part_ids) {
  int num_gpus = num_machines * num_gpus_per_machine;

  std::vector<std::vector<int64_t>> src_id_buffers(num_gpus);
  std::vector<std::vector<int64_t>> dst_id_buffers(num_gpus);

  int64_t* src_ptr = (int64_t*) global_src->data;
  int64_t* dst_ptr = (int64_t*) global_dst->data;
  int64_t* part_ptr = (int64_t*) global_src_part_ids->data;

  for (int i = 0; i < global_src->shape[0]; i++) {
    auto src_id = *src_ptr++;
    auto dst_id = *dst_ptr++;
    auto part_id = *part_ptr++;

    int global_gpu_rank = part_id * num_gpus_per_machine + src_id % num_gpus_per_machine;
    src_id_buffers[global_gpu_rank].push_back(src_id);
    dst_id_buffers[global_gpu_rank].push_back(dst_id);
  }

  auto global_src_list = std::vector<IdArray>();
  auto global_dst_list = std::vector<IdArray>();

  for (int i = 0; i < num_gpus; i++) {
    global_src_list.push_back(IdArray::FromVector(src_id_buffers[i]));
    global_dst_list.push_back(IdArray::FromVector(dst_id_buffers[i]));
  }

  return std::make_pair(global_src_list, global_dst_list);
}

}
}
