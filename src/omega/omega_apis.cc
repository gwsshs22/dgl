#include <iostream>

#include <dgl/runtime/container.h>
#include "../c_api_common.h"

#include "distributed_block.h"
#include "distributed_sampling.h"
#include "partition_request.h"
#include "sampling.h"
#include "trace_gen_helper.h"

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

DGL_REGISTER_GLOBAL("omega.omega_apis._CAPI_DGLOmegaToBlock")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    const HeteroGraphRef empty_graph_ref = args[0];
    IdArray u = args[1];
    IdArray v = args[2];
    IdArray dst_ids = args[3];
    IdArray src_ids = args[4];

    runtime::List<runtime::ObjectRef> ret_list;
    auto ret = omega::ToBlock(empty_graph_ref, u, v, dst_ids, src_ids);
    ret_list.push_back(HeteroGraphRef(ret.first));
    ret_list.push_back(runtime::Value(MakeValue(ret.second)));
    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("omega.omega_apis._CAPI_DGLOmegaPartitionRequest")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int num_machines = args[0];
    int num_gpus_per_machine = args[1];
    IdArray nid_partitions = args[2];
    IdArray num_assigned_targets_per_gpu = args[3];
    IdArray target_gnids = args[4];
    IdArray src_gnids = args[5];
    IdArray dst_gnids = args[6];

    auto ret = omega::PartitionRequest(
      num_machines,
      num_gpus_per_machine,
      nid_partitions,
      num_assigned_targets_per_gpu,
      target_gnids,
      src_gnids,
      dst_gnids);

    runtime::List<runtime::Value> ret_list;
    for (const auto& s : ret.first) {
      ret_list.push_back(runtime::Value(MakeValue(s)));
    }

    for (const auto& d : ret.second) {
      ret_list.push_back(runtime::Value(MakeValue(d)));
    }

    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("omega.omega_apis._CAPI_DGLOmegaTraceGenHelper")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int first_new_gnid = args[0];
    IdArray infer_target_mask = args[1];
    IdArray batch_local_ids = args[2];
    IdArray u = args[3];
    IdArray v = args[4];
    IdArray u_in_partitions = args[5];
    IdArray v_in_partitions = args[6];

    auto ret = omega::TraceGenHelper(
      first_new_gnid,
      infer_target_mask,
      batch_local_ids,
      u,
      v,
      u_in_partitions,
      v_in_partitions);

    runtime::List<runtime::Value> ret_list;
    for (int i = 0; i < 3; i++) {
      ret_list.push_back(runtime::Value(MakeValue(ret[i])));
    }

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

DGL_REGISTER_GLOBAL("omega.omega_apis._CAPI_DGLOmegaSampleEdges")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    IdArray target_gnids = args[0];
    IdArray src_gnids = args[1];
    IdArray dst_gnids = args[2];
    IdArray fanouts_array = args[3];
    const auto& fanouts = fanouts_array.ToVector<int64_t>();

    auto ret = omega::SampleEdges(
      target_gnids,
      src_gnids,
      dst_gnids,
      fanouts);

    runtime::List<runtime::Value> ret_list;
    for (auto p : ret) {
      ret_list.push_back(runtime::Value(MakeValue(p.first)));
      ret_list.push_back(runtime::Value(MakeValue(p.second)));
    }

    *rv = ret_list;
  });

}
