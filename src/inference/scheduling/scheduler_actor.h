#pragma once

#include <queue>
#include <map>
#include <dgl/inference/common.h>

#include "scheduling.h"
#include "scheduled_batch.h"

namespace dgl {
namespace inference {

class scheduler_actor : public caf::event_based_actor, Scheduler {

 public:
  scheduler_actor(caf::actor_config& config,
                  const caf::strong_actor_ptr& exec_ctl_actor_ptr,
                  caf::actor result_collect_actor,
                  ParallelizationType parallelization_type,
                  bool using_precomputed_aggs,
                  int num_nodes,
                  int num_devices_per_node);

 private:
  caf::behavior make_behavior() override;

  void LocalInitialize(int batch_id, int node_rank, const BatchInput& batch_input) override;

  void LocalExecute(TaskType task_type, int batch_id, int node_rank, int local_rank) override;

  void LocalFetchResult(int batch_id, int node_rank, int local_rank) override;

  void BroadcastInitialize(int batch_id, const BatchInput& batch_input) override;

  void BroadcastExecute(TaskType task_type, int batch_id) override;

  void BroadcastFetchResult(int batch_id) override;

  void ReportResult(int request_id, NDArray result) override;

  caf::actor exec_ctl_actor_;
  caf::actor result_collect_actor_;
  std::shared_ptr<SchedulingPolicy> policy_;
  int request_id_counter_ = 0;
};

}
}
