#include "executor_control_actor.h"

#include <algorithm>

namespace dgl {
namespace inference {

executor_control_actor::executor_control_actor(caf::actor_config& config)
    : event_based_actor(config) {
}

caf::behavior executor_control_actor::make_behavior() {
  return initializing();
}

caf::behavior executor_control_actor::initializing() {
  return {
    [&](caf::set_atom) { // From scheduler
      scheduler_connected_ = true;
      scheduler_actor_ = caf::actor_cast<caf::actor>(current_sender());
      TryRunning();
    },
    [&](caf::initialized_atom, const caf::strong_actor_ptr& executor_ptr, int executor_rank, int num_nodes) { // From executors
      assert(num_nodes >= 1);
      assert(executor_rank >= 0);
      assert(executor_rank < num_nodes);
      for (auto& p : pending_executors_) {
        assert(p.second != executor_rank);
      }

      if (num_nodes_ < 0) {
        num_nodes_ = num_nodes;
      } else {
        assert(num_nodes == num_nodes_);
      }

      pending_executors_.push_back(std::make_pair(executor_ptr, executor_rank));
      TryRunning();
    }
  };
}

void executor_control_actor::TryRunning() {
  if ((pending_executors_.size() == num_nodes_) && scheduler_connected_) {
    std::sort(pending_executors_.begin(), pending_executors_.end(), [](const auto& e1, const auto& e2) { return e1.second < e2.second; });
    for (auto const& pe : pending_executors_) {
      executors_.emplace_back(caf::actor_cast<caf::actor>(pe.first));
    }

    become(running());
    pending_executors_.clear();
    ReportToInitMon(*this, "exec_ctrl", 0, 1);
  }
}

caf::behavior executor_control_actor::running() {
  return {
    [&](caf::init_atom, int batch_id, const NDArray& new_gnids, const NDArray& src_gnids, const NDArray& dst_gnids) {
      send(executors_[0], caf::init_atom_v, batch_id, new_gnids, src_gnids, dst_gnids);
      
      for (int i = 1; i < num_nodes_; i++) {
        send(executors_[i], caf::init_atom_v, batch_id);
      }
    },
    [&](caf::exec_atom, TaskType task_type, int batch_id) {
      for (int i = 0; i < num_nodes_; i++) {
        send(executors_[i], caf::exec_atom_v, task_type, batch_id);
      }
    },
    [&](caf::done_atom, TaskType task_type, int batch_id, int rank) {
      auto p = std::make_pair(task_type, batch_id);
      auto it = done_task_counter_.find(p);

      if (it == done_task_counter_.end()) {
        it = done_task_counter_.emplace(p, 1).first;
      } else {
        ((*it).second)++;
      }

      if (it->second == num_nodes_) {
        done_task_counter_.erase(it);
        send(scheduler_actor_, caf::done_atom_v, task_type, batch_id);
      }
    }
  };
}

}
}
