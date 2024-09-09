#include <iostream>
#include <chrono>

#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"

#include "sampler_v2.h"

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("omega.sampler_v2._CAPI_DGLOmegaCreateSampler")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    const int num_machines = args[0];
    const int machine_rank = args[1];
    const int num_gpus_per_machine_in_group = args[2];
    const int gpu_rank_in_group = args[3];
    const int local_rank = args[4];
    const IdArray nid_partitions_arr = args[5];
    const std::vector<int64_t> nid_partitions = nid_partitions_arr.ToVector<int64_t>();
    const int num_layers = args[6];
    const IdArray fanouts_arr = args[7];
    const std::vector<int64_t> fanouts = fanouts_arr.ToVector<int64_t>();
    const bool is_cgp = args[8];
    const int recom_threshold = args[9];
    const PackedFunc pull_fn = args[10];
    const PackedFunc dist_sampling_fn = args[11];
    const PackedFunc pe_recom_policy_fn = args[12];
    const PackedFunc all_gather_fn = args[13];
    const PackedFunc dist_edges_fn = args[14];
    const PackedFunc filter_cached_id_fn = args[15];
    const HeteroGraphRef empty_graph_ref = args[16];
    const HeteroGraphRef local_graph_ref = args[17];
    const IdArray local_graph_global_id_mapping = args[18];
    const List<Value> local_data_names = args[19];
    const List<Value> local_data_tensors = args[20];
    const IdArray in_degrees = args[21];
    const IdArray out_degrees = args[22];
    const IdArray cached_id_map = args[23];
    const IdArray gnid_to_local_id_mapping = args[24];

    std::unordered_map<std::string, NDArray> local_data_store;
    CHECK_EQ(local_data_names.size(), local_data_tensors.size());
    for (int i = 0; i < local_data_names.size(); i++) {
      local_data_store[local_data_names[i]->data] = local_data_tensors[i]->data;
    }

    omega::SamplingExecutorV2Ref ref = omega::SamplingExecutorV2Ref(std::make_shared<omega::SamplingExecutorV2>(
      num_machines,
      machine_rank,
      num_gpus_per_machine_in_group,
      gpu_rank_in_group,
      local_rank,
      nid_partitions,
      num_layers,
      fanouts,
      is_cgp,
      recom_threshold,
      pull_fn,
      dist_sampling_fn,
      pe_recom_policy_fn,
      all_gather_fn,
      dist_edges_fn,
      filter_cached_id_fn,
      empty_graph_ref,
      local_graph_ref,
      local_graph_global_id_mapping,
      local_data_store,
      in_degrees,
      out_degrees,
      cached_id_map,
      gnid_to_local_id_mapping
    ));

    *rv = ref;
  });

DGL_REGISTER_GLOBAL("omega.sampler_v2._CAPI_DGLOmegaSampleBlocksDp")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    omega::SamplingExecutorV2Ref executor = args[0];
    const int batch_id = args[1];
    const IdArray target_gnids = args[2];
    const IdArray src_gnids = args[3];
    const IdArray dst_gnids = args[4];

    auto sampling_ret = executor->SampleBlocksDp(batch_id, target_gnids, src_gnids, dst_gnids);

    List<runtime::Value> ret_list;

    for (const auto& p : sampling_ret.first) {
      ret_list.push_back(runtime::Value(MakeValue(HeteroGraphRef(p.first))));
      ret_list.push_back(runtime::Value(MakeValue(p.second)));
    }

    for (const auto& p : sampling_ret.second) {
      ret_list.push_back(runtime::Value(MakeValue(p)));
    }

    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("omega.sampler_v2._CAPI_DGLOmegaSampleBlocksPrecoms")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    omega::SamplingExecutorV2Ref executor = args[0];
    const int batch_id = args[1];
    const IdArray target_gnids = args[2];
    const IdArray src_gnids = args[3];
    const IdArray dst_gnids = args[4];
    const IdArray new_lhs_ids_prefix = args[5];

    auto sampling_ret = executor->SampleBlocksPrecoms(batch_id, target_gnids, src_gnids, dst_gnids, new_lhs_ids_prefix);

    List<runtime::Value> ret_list;

    for (const auto& p : sampling_ret.first) {
      ret_list.push_back(runtime::Value(MakeValue(HeteroGraphRef(p.first))));
      ret_list.push_back(runtime::Value(MakeValue(p.second)));
    }

    for (const auto& p : sampling_ret.second) {
      ret_list.push_back(runtime::Value(MakeValue(p)));
    }

    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("omega.sampler_v2._CAPI_DGLOmegaSampleBlocksDpPrecomsWithRecom")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    omega::SamplingExecutorV2Ref executor = args[0];
    const int batch_id = args[1];
    const IdArray target_gnids = args[2];
    const IdArray src_gnids = args[3];
    const IdArray dst_gnids = args[4];

    auto ret = executor->SampleBlocksDpPrecomsWithRecom(batch_id, target_gnids, src_gnids, dst_gnids);
    List<runtime::Value> ret_list;

    for (const auto& p : ret.first.first) {
      ret_list.push_back(runtime::Value(MakeValue(HeteroGraphRef(p.first))));
      ret_list.push_back(runtime::Value(MakeValue(p.second)));
    }

    for (const auto& p : ret.first.second) {
      ret_list.push_back(runtime::Value(MakeValue(p)));
    }

    ret_list.push_back(runtime::Value(MakeValue(ret.second)));
    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("omega.sampler_v2._CAPI_DGLOmegaSampleBlocksCgpWithRecom")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    omega::SamplingExecutorV2Ref executor = args[0];
    const int batch_id = args[1];
    const IdArray target_gnids = args[2];
    const IdArray src_gnids = args[3];
    const IdArray dst_gnids = args[4];
    const IdArray new_lhs_ids_prefix = args[5];

    auto ret = executor->SampleBlocksCgpWithRecom(batch_id, target_gnids, src_gnids, dst_gnids, new_lhs_ids_prefix);
    List<runtime::Value> ret_list;

    for (const auto& p : std::get<0>(ret).first) {
      ret_list.push_back(runtime::Value(MakeValue(HeteroGraphRef(p.first))));
      ret_list.push_back(runtime::Value(MakeValue(p.second)));
    }

    for (const auto& p : std::get<0>(ret).second) {
      ret_list.push_back(runtime::Value(MakeValue(p)));
    }

    ret_list.push_back(runtime::Value(MakeValue(std::get<1>(ret))));
    ret_list.push_back(runtime::Value(MakeValue(std::get<2>(ret))));
    *rv = ret_list;
  });

}
