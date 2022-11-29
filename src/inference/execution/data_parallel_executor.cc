#include "executor_actor.h"

#include "./sampling/sampling_actor.h"
#include "../process/process_control_actor.h"

#include "mem_utils.h"

namespace dgl {
namespace inference {

namespace {

void direct_fetch_result_fn(caf::blocking_actor* self,
                            int batch_id) {
  auto src_arr = LoadFromSharedMemory(batch_id, "result");
  NDArray copied = NDArray::Empty(
      std::vector<int64_t>(src_arr->shape, src_arr->shape + src_arr->ndim),
      src_arr->dtype,
      DLContext{kDLCPU, 0});

  copied.CopyFrom(src_arr);
  self->receive([=](caf::get_atom) {
    return copied;
  });
};

void fetch_result_fn(caf::blocking_actor* self,
                     const caf::actor& mpi_actor,
                     int batch_id) {
  auto result = LoadFromSharedMemory(batch_id, "result");
  auto rh2 = self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, 0, result, CreateMpiTag(batch_id, TaskType::kFetchResult));
  receive_result<bool>(rh2);

  self->receive([](caf::get_atom) { });
}

void write_traces_fn(caf::blocking_actor* self,
                     const caf::actor& trace_actor,
                     const std::string& result_dir,
                     int num_devices_per_node,
                     int node_rank,
                     const caf::actor& gnn_executor_group,
                     const std::vector<caf::actor>& samplers,
                     caf::response_promise rp) {
  auto rh = self->request(trace_actor, caf::infinite, caf::write_trace_atom_v);
  receive_result<bool>(rh);

  for (int i = 0; i < samplers.size(); i++) {
    auto rh = self->request(samplers[i], caf::infinite, caf::write_trace_atom_v);
    receive_result<bool>(rh);
  }

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

data_parallel_executor::data_parallel_executor(caf::actor_config& config,
                                               caf::strong_actor_ptr exec_ctl_actor_ptr,
                                               caf::strong_actor_ptr mpi_actor_ptr,
                                               caf::strong_actor_ptr trace_actor_ptr,
                                               int node_rank,
                                               int num_nodes,
                                               int num_backup_servers,
                                               int num_devices_per_node,
                                               int num_samplers_per_node,
                                               std::string result_dir,
                                               bool collect_stats)
    : executor_actor(config,
                     exec_ctl_actor_ptr,
                     mpi_actor_ptr,
                     trace_actor_ptr,
                     node_rank,
                     num_nodes,
                     num_backup_servers,
                     num_devices_per_node,
                     result_dir,
                     collect_stats,
                     num_samplers_per_node) {
  auto self_ptr = caf::actor_cast<caf::strong_actor_ptr>(this);
  for (int i = 0; i < num_samplers_per_node; i++) {
    samplers_.emplace_back(spawn<sampling_actor, caf::linked + caf::monitored>(self_ptr, i));
    sampler_running_.push_back(false);
  }
}

void data_parallel_executor::Sampling(int batch_id, int, int) {
  int idle_sampler_rank = -1;
  for (int i = 0; i < sampler_running_.size(); i++) {
    if (!sampler_running_[i]) {
      idle_sampler_rank = i;
      break;
    }
  }

  CHECK(idle_sampler_rank != -1);
  sampler_running_[idle_sampler_rank] = true;
  batch_id_to_sampler_rank_[batch_id] = idle_sampler_rank;
  auto sampling_task = spawn(sampling_fn, samplers_[idle_sampler_rank], batch_id);

  RequestAndReportTaskDone(sampling_task, TaskType::kSampling, batch_id, [&, idle_sampler_rank]() {
    sampler_running_[idle_sampler_rank] = false;
  });
}

void data_parallel_executor::Compute(int batch_id, int local_rank, int, int) {
  auto compute_task = spawn(gnn_execute_fn, gnn_executor_group_, batch_id, gnn_executor_request_type::kComputeRequestType, local_rank, /* param0 */ -1);
  RequestAndReportTaskDone(compute_task, TaskType::kCompute, batch_id);
}

void data_parallel_executor::DirectFetchResult(int batch_id,
                                               int,
                                               caf::response_promise rp) {
  auto task = spawn(direct_fetch_result_fn, batch_id);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=](NDArray result) mutable {
      rp.deliver(std::vector<NDArray>({result}));
    },
    [=](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << "DirectFetchResult (batch_id=" << batch_id << "): " << caf::to_string(err) << std::endl;
    });
}

void data_parallel_executor::FetchResult(int batch_id, int) {
  auto task = spawn(fetch_result_fn, mpi_actor_, batch_id);
  request(task, caf::infinite, caf::get_atom_v).then(
    [=]() { },
    [=](caf::error& err) {
      // TODO: error handling
      caf::aout(this) << "FetchResult (batch_id=" << batch_id << "): " << caf::to_string(err) << std::endl;
    });
}

void data_parallel_executor::Cleanup(int batch_id, int gpu_local_rank, int) {
  auto it = batch_id_to_sampler_rank_.find(batch_id);
  CHECK(it != batch_id_to_sampler_rank_.end());
  int sampler_rank = it->second;
  batch_id_to_sampler_rank_.erase(it);
  send(samplers_[sampler_rank], caf::cleanup_atom_v, batch_id);
  send(gnn_executor_group_,
       caf::exec_atom_v,
       batch_id,
       static_cast<int>(gnn_executor_request_type::kCleanupRequestType),
       gpu_local_rank,
       /* param0 */ -1);

  object_storages_.erase(batch_id);
  ReportTaskDone(TaskType::kCleanup, batch_id);
}

void data_parallel_executor::WriteExecutorTraces(caf::response_promise rp) {
  spawn(write_traces_fn, trace_actor_, result_dir_, num_devices_per_node_, node_rank_, gnn_executor_group_, samplers_, rp);
}

}
}
