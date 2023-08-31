/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/sampler.h
 * \brief 
 */
#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>

#include <dmlc/blockingconcurrentqueue.h>
#include <dmlc/thread_group.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/object.h>
#include <dgl/runtime/packed_func.h>

namespace dgl {
namespace omega {

using namespace dgl::runtime;

class SamplingExecutor : public Object {

 public:
  struct Task {
    const int batch_id = -1;
    const IdArray target_gnids;
    const IdArray src_gnids;
    const IdArray dst_gnids;
    const PackedFunc callback;

    Task(const int b, const IdArray& t, const IdArray& s, const IdArray& d, const PackedFunc& c)
      : batch_id(b), target_gnids(t), src_gnids(s), dst_gnids(d), callback(c) {}

  };

  typedef dmlc::moodycamel::BlockingConcurrentQueue<std::shared_ptr<Task>> QueueT;

  SamplingExecutor(
    const int num_threads,
    const int num_machines,
    const int machine_rank,
    const int num_gpus_per_machine_in_group,
    const int local_gpu_rank_in_group,
    const std::vector<int64_t>& nid_partitions,
    const std::string& exec_mode,
    const bool use_precoms,
    const int num_layers,
    const std::vector<int64_t>& fanouts,
    const PackedFunc& pull_fn,
    const PackedFunc& dist_sampling_fn,
    const HeteroGraphRef& empty_graph_ref,
    const HeteroGraphRef& local_graph_ref,
    const IdArray& local_graph_global_id_mapping,
    const std::unordered_map<std::string, NDArray>& local_data_store);

  void Enqueue(std::shared_ptr<Task>&& task);

  void Shutdown();

  std::pair<HeteroGraphPtr, IdArray> ToBlock_(
    const IdArray& target_gnids, const IdArray& src_gnids, const IdArray& dst_gnids) const;

  std::tuple<HeteroGraphPtr, IdArray, IdArray> ToDistributedBlock_(
    const IdArray& target_gnids, const IdArray& src_gnids, const IdArray& dst_gnids) const;

  int64_t GetNumLocalTargets(int64_t num_targets) const;

  NDArray Pull(const std::string& name, const IdArray& id_arr) const;

  std::pair<std::vector<IdArray>, std::vector<IdArray>> DistSampling(
    const IdArray& seeds, const int fanout) const;

  const inline int num_threads() const {
    return num_threads_;
  };

  const inline int machine_rank() const {
    return machine_rank_;
  };

  const inline int num_machines() const {
    return num_machines_;
  };

  const inline int num_gpus_per_machine_in_group() const {
    return num_gpus_per_machine_in_group_;
  };

  const inline int local_gpu_rank_in_group() const {
    return local_gpu_rank_in_group_;
  };

  const inline std::vector<int64_t>& nid_partitions() const {
    return nid_partitions_;
  };

  const inline std::string& exec_mode() const {
    return exec_mode_;
  };

  const inline bool use_precoms() const {
    return use_precoms_;
  }

  const inline int num_layers() const {
    return num_layers_;
  };

  const inline std::vector<int64_t>& fanouts() const {
    return fanouts_;
  };

  const inline std::vector<int64_t>& cgp_fanouts() const {
    return cgp_fanouts_;
  }

  const inline bool full_sampling() const {
    return full_sampling_;
  }

  const inline IdArray& null_array() const {
    return null_array_;
  }

  static constexpr const char* _type_key = "omega.sampler.SamplingExecutor";
  DGL_DECLARE_OBJECT_TYPE_INFO(SamplingExecutor, runtime::Object);

 private:

  IdArray GetPartIds(const IdArray& src_gnids) const;

  PackedFunc SendRemotePullReqs(
    const std::string& name,
    const std::vector<int>& part_ids,
    const std::vector<IdArray>& local_nids_list) const;

  void CopyLocalData(
    NDArray& ret_tensor,
    const NDArray& local_tensor,
    const IdArray& local_nids,
    const IdArray& orig_pos,
    const int row_size) const;

  void CopyFetchedData(
    NDArray& ret_tensor,
    const NDArray& fetched_tensor,
    const IdArray& orig_pos,
    const int row_size) const;

  PackedFunc SendRemoteDistSamplingReqs(
    const int fanout,
    const std::vector<int>& part_ids,
    const std::vector<IdArray>& local_nids_list) const;
  
  std::pair<IdArray, IdArray> LocalSampling(const IdArray& seeds, const int fanout) const;

  dmlc::ThreadGroup thread_group_;
  std::shared_ptr<QueueT> queue_;
  const int num_threads_;
  const int num_machines_;
  const int machine_rank_;
  const int num_gpus_per_machine_in_group_;
  const int local_gpu_rank_in_group_;
  const std::vector<int64_t> nid_partitions_;
  const std::string exec_mode_;
  const bool use_precoms_;
  const int num_layers_;
  const std::vector<int64_t> fanouts_;
  const std::vector<int64_t> cgp_fanouts_;
  const bool full_sampling_;
  const PackedFunc pull_fn_;
  const PackedFunc dist_sampling_fn_;
  const HeteroGraphRef empty_graph_ref_;
  const HeteroGraphRef local_graph_ref_;
  const IdArray local_graph_global_id_mapping_;
  const std::unordered_map<std::string, NDArray> local_data_store_;
  const IdArray null_array_;
};

DGL_DEFINE_OBJECT_REF(SamplingExecutorRef, SamplingExecutor);

}
}
