#include <iostream>

#include <dgl/runtime/container.h>
#include "../c_api_common.h"

#include "distributed_block.h"
#include "distributed_sampling.h"

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("omega.omega_apis._CAPI_DGLOmegaToDistributedBlocks")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int num_machines = args[0];
    int machine_rank = args[1];
    int num_gpus_per_machine = args[2];
    IdArray target_gnids = args[3];
    IdArray src_gnids = args[4];
    IdArray src_part_ids = args[5];
    IdArray dst_gnids = args[6];

    auto results = omega::ToDistributedBlocks(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        target_gnids,
        src_gnids,
        src_part_ids,
        dst_gnids);

    CHECK_EQ(num_gpus_per_machine, results.first.size());
    CHECK_EQ(num_gpus_per_machine, results.second.size());

    List<runtime::Value> ret_list;
    for (int i = 0; i < num_gpus_per_machine; i++) {
      ret_list.push_back(runtime::Value(MakeValue(HeteroGraphRef(results.first[i]))));
      ret_list.push_back(runtime::Value(MakeValue(results.second[i])));
    }

    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("omega.omega_apis._CAPI_DGLOmegaToDistributedBlock")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int num_machines = args[0];
    int machine_rank = args[1];
    int num_gpus_per_machine = args[2];
    int local_gpu_idx = args[3];
    IdArray target_gnids = args[4];
    IdArray src_gnids = args[5];
    IdArray src_part_ids = args[6];
    IdArray dst_gnids = args[7];

    auto result = omega::ToDistributedBlock(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        local_gpu_idx,
        target_gnids,
        src_gnids,
        src_part_ids,
        dst_gnids);

    List<runtime::Value> ret_list;
    ret_list.push_back(runtime::Value(MakeValue(HeteroGraphRef(result.first))));
    ret_list.push_back(runtime::Value(MakeValue(result.second)));

    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("omega.dist_sample._CAPI_DGLOmegaSplitLocalEdges")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int num_machines = args[0];
    int machine_rank = args[1];
    int num_gpus_per_machine = args[2];
    int gpu_rank = args[3];
    IdArray global_src = args[4];
    IdArray global_dst = args[5];
    IdArray global_src_part_ids = args[6];

    auto ret = omega::SplitLocalEdges(
      num_machines,
      machine_rank,
      num_gpus_per_machine,
      gpu_rank, 
      global_src, 
      global_dst, 
      global_src_part_ids);

    int num_gpus = num_machines * num_gpus_per_machine;
    runtime::List<runtime::Value> ret_list;
    for (int i = 0; i < num_gpus; i++) {
      ret_list.push_back(runtime::Value(MakeValue(ret.first[i])));
    }

    for (int i = 0; i < num_gpus; i++) {
      ret_list.push_back(runtime::Value(MakeValue(ret.second[i])));
    }

    *rv = ret_list;
  });

}
