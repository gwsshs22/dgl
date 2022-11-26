#include "executor_actor.h"

#include <dgl/array.h>

#include "./sampling/sampling_actor.h"
#include "../process/process_control_actor.h"

#include "mem_utils.h"

namespace dgl {
namespace inference {

namespace {

void direct_fetch_result_fn(caf::blocking_actor* self,
                            int num_devices_per_node,
                            int batch_id) {
  auto results = std::vector<NDArray>();
  for (int local_rank = 0; local_rank < num_devices_per_node; local_rank++) {
    auto src_arr = LoadFromSharedMemory(batch_id, "g" + std::to_string(local_rank) + "_result");
    NDArray copied = NDArray::Empty(
        std::vector<int64_t>(src_arr->shape, src_arr->shape + src_arr->ndim),
        src_arr->dtype,
        DLContext{kDLCPU, 0});

    copied.CopyFrom(src_arr);
    results.push_back(copied);
  }

  self->receive([=](caf::get_atom) {
    return results;
  });
};

void fetch_result_fn(caf::blocking_actor* self,
                     const caf::actor& mpi_actor,
                     const int node_rank,
                     int num_devices_per_node,
                     int batch_id) {
  auto rhs = std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>>();
  for (int local_rank = 0; local_rank < num_devices_per_node; local_rank++) {
    auto result = LoadFromSharedMemory(batch_id, "g" + std::to_string(local_rank) + "_result");
    rhs.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, 0, result, CreateMpiTag(batch_id, TaskType::kFetchResult, node_rank, local_rank)));
  }

  for (int local_rank = 0; local_rank < num_devices_per_node; local_rank++) {
    receive_result<bool>(rhs[local_rank]);
  }

  self->receive([](caf::get_atom) { });
}

void write_traces_fn(caf::blocking_actor* self,
                     const std::string& result_dir,
                     int num_devices_per_node,
                     int node_rank,
                     const caf::actor& gnn_executor_group,
                     const caf::actor& sampler,
                     caf::response_promise rp) {
  WriteTraces(result_dir, node_rank);

  auto rh = self->request(sampler, caf::infinite, caf::write_trace_atom_v);
  receive_result<bool>(rh);

  for (int i = 0; i < num_devices_per_node; i++) {
    auto rh = self->request(gnn_executor_group,
                            caf::infinite,
                            caf::exec_atom_v,
                            -1,
                            static_cast<int>(gnn_executor_request_type::kWriteTraces),
                            i,
                            -1);
    receive_result<bool>(rh);
  }

  rp.deliver(true);  
}

}

vertex_cut_executor::vertex_cut_executor(caf::actor_config& config,
                                         caf::strong_actor_ptr exec_ctl_actor_ptr,
                                         caf::strong_actor_ptr mpi_actor_ptr,
                                         int node_rank,
                                         int num_nodes,
                                         int num_backup_servers,
                                         int num_devices_per_node,
                                         std::string result_dir,
                                         bool collect_stats,
                                         bool using_precomputed_aggs)
    : executor_actor(config,
                     exec_ctl_actor_ptr,
                     mpi_actor_ptr,
                     node_rank,
                     num_nodes,
                     num_backup_servers,
                     num_devices_per_node,
                     result_dir,
                     collect_stats,
                     1),
      using_precomputed_aggs_(using_precomputed_aggs) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  sampler_ = spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, -1);
}

void vertex_cut_executor::Sampling(int batch_id, int) {
  // OPTIMIZATION TODO: Sampling in c++.
  auto sampling_task = spawn(sampling_fn, sampler_, batch_id);
  RequestAndReportTaskDone(sampling_task, TaskType::kSampling, batch_id);
}

void vertex_cut_executor::PrepareInput(int batch_id, int) {
  ReportTaskDone(TaskType::kPrepareInput, batch_id);
}

void vertex_cut_executor::Compute(int batch_id, int, int, int) {
  auto compute_task = spawn(gnn_broadcast_execute_fn, gnn_executor_group_, batch_id, gnn_executor_request_type::kComputeRequestType, /* param0 */ -1);
  RequestAndReportTaskDone(compute_task, TaskType::kCompute, batch_id);
}

void vertex_cut_executor::PrepareAggregations(int batch_id, int) {
  ReportTaskDone(TaskType::kPrepareAggregations, batch_id);
}

void vertex_cut_executor::RecomputeAggregations(int batch_id, int) {
  ReportTaskDone(TaskType::kRecomputeAggregations, batch_id);
}

void vertex_cut_executor::ComputeRemaining(int batch_id, int) {
  ReportTaskDone(TaskType::kComputeRemaining, batch_id);
}

void vertex_cut_executor::DirectFetchResult(int batch_id, int, caf::response_promise rp) {
  auto task = spawn(direct_fetch_result_fn, num_devices_per_node_, batch_id);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=](std::vector<NDArray> result) mutable {
      rp.deliver(result);
      this->Cleanup(batch_id);
    },
    [=](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << "DirectFetchResult (batch_id=" << batch_id << "): " << caf::to_string(err) << std::endl;
    });
}

void vertex_cut_executor::FetchResult(int batch_id, int) {
  auto task = spawn(fetch_result_fn, mpi_actor_, node_rank_, num_devices_per_node_, batch_id);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { this->Cleanup(batch_id); },
    [=](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << "FetchResult (batch_id=" << batch_id << "): " << caf::to_string(err) << std::endl;
    });
}

void vertex_cut_executor::Cleanup(int batch_id) {
  send(sampler_, caf::cleanup_atom_v, batch_id);
  send(gnn_executor_group_,
       caf::broadcast_exec_atom_v,
       batch_id,
       static_cast<int>(gnn_executor_request_type::kCleanupRequestType),
       /* param0 */ -1);

  object_storages_.erase(batch_id);
}

void vertex_cut_executor::WriteExecutorTraces(caf::response_promise rp) {
  spawn(write_traces_fn, result_dir_, num_devices_per_node_, node_rank_, gnn_executor_group_, sampler_, rp);
}

}
}
