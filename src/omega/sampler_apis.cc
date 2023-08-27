#include <iostream>

#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"

#include "sampler.h"

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("omega.sampler._CAPI_DGLCreateSamplingExecutor")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    const int num_threads = args[0];
    const int num_machines = args[1];
    const int machine_rank = args[2];
    const int num_gpus_per_machine_in_group = args[3];
    const int local_gpu_rank_in_group = args[4];
    const IdArray nid_partitions_arr = args[5];
    const std::vector<int64_t> nid_partitions = nid_partitions_arr.ToVector<int64_t>();
    const std::string exec_mode = args[6];
    const bool use_precoms = args[7];
    const int num_layers = args[8];
    const IdArray fanouts_arr = args[9];
    const std::vector<int64_t> fanouts = fanouts_arr.ToVector<int64_t>();
    const PackedFunc pull_fn = args[10];
    const PackedFunc dist_sampling_fn = args[11];
    const HeteroGraphRef empty_graph_ref = args[12];
    const HeteroGraphRef local_graph_ref = args[13];
    const IdArray local_graph_global_id_mapping = args[14];
    const List<Value> local_data_names = args[15];
    const List<Value> local_data_tensors = args[16];

    std::unordered_map<std::string, NDArray> local_data_store;
    CHECK_EQ(local_data_names.size(), local_data_tensors.size());
    for (int i = 0; i < local_data_names.size(); i++) {
      local_data_store[local_data_names[i]->data] = local_data_tensors[i]->data;
    }

    omega::SamplingExecutorRef ref = omega::SamplingExecutorRef(std::make_shared<omega::SamplingExecutor>(
      num_threads,
      num_machines,
      machine_rank,
      num_gpus_per_machine_in_group,
      local_gpu_rank_in_group,
      nid_partitions,
      exec_mode,
      use_precoms,
      num_layers,
      fanouts,
      pull_fn,
      dist_sampling_fn,
      empty_graph_ref,
      local_graph_ref,
      local_graph_global_id_mapping,
      local_data_store
    ));

    *rv = ref;
  });


DGL_REGISTER_GLOBAL("omega.sampler._CAPI_DGLEnqueue")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    omega::SamplingExecutorRef executor = args[0];
    const int batch_id = args[1];
    const IdArray target_gnids = args[2];
    const IdArray src_gnids = args[3];
    const IdArray dst_gnids = args[4];
    const PackedFunc callback = args[5];

    executor->Enqueue(std::make_shared<omega::SamplingExecutor::Task>(
      batch_id,
      std::move(target_gnids),
      std::move(src_gnids),
      std::move(dst_gnids),
      std::move(callback)
    ));

  });

DGL_REGISTER_GLOBAL("omega.sampler._CAPI_DGLShutdown")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    omega::SamplingExecutorRef executor = args[0];
    executor->Shutdown();
  });
}
