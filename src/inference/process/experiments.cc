#include "experiments.h"

namespace dgl {
namespace inference {

void input_feader_fn(caf::blocking_actor* self,
                     const caf::actor& scheduler,
                     const std::string& input_trace_dir,
                     int num_warmup_reqs,
                     int num_reqs) {

  
  auto new_gnids_vec = std::vector<NDArray>();
  auto new_features_vec = std::vector<NDArray>();
  auto src_gnids_vec = std::vector<NDArray>();
  auto dst_gnids_vec = std::vector<NDArray>();

  auto cpu_context = DLContext { kDLCPU, 0 };

  int num_inputs = 10;
  int feature_size = 256;
  NDArray new_features = NDArray::Empty({num_inputs, feature_size}, DLDataType{kDLFloat, 32, 1}, cpu_context);
  float* ptr = (float*)new_features->data;
  for (int i = 0; i < num_inputs * feature_size; i++) {
    *ptr++ = (float)(i + 1) / (float)feature_size;
  }

  NDArray new_gnids = NDArray::FromVector(std::vector<int64_t>{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, cpu_context);
  
  NDArray src_gnids = NDArray::FromVector(std::vector<int64_t>{ 10, 10, 10, 10, 10,  3,  4,  0,  7,  9,  0, 12, 13, 14, 15, 9  }, cpu_context);
  NDArray dst_gnids = NDArray::FromVector(std::vector<int64_t>{  0,  1, 12, 13, 11, 10, 12, 13, 10, 10, 14, 15, 16, 17, 18, 19 }, cpu_context);

  for (int i = 0; i < num_warmup_reqs; i++) {
    auto rh = self->request(scheduler, caf::infinite, caf::enqueue_atom_v, new_gnids, new_features, src_gnids, dst_gnids);
    receive_result<int>(rh);
  }

  // Wait for warmup done.
  std::cout << "input_feader wait start message" << std::endl;
  self->receive([](caf::start_atom){});

  for (int i = 0; i < num_reqs; i++) {
    auto rh = self->request(scheduler, caf::infinite, caf::enqueue_atom_v, new_gnids, new_features, src_gnids, dst_gnids);
    receive_result<int>(rh);
  }

  // Wait for all done
  self->receive([](caf::done_atom){});
}

caf::behavior result_receiver_fn(caf::stateful_actor<result_receiver_state>* self,
                                 int num_warmup_reqs,
                                 int num_reqs) {
  self->state.num_warmups_reqs = num_warmup_reqs;
  self->state.num_reqs = num_reqs;
  self->state.num_done_warmups_reqs = 0;
  self->state.num_done_reqs = 0;
  self->state.warmup_finished = false;
  self->state.warmup_waiting = false;
  self->state.finished = false;
  self->state.waiting = false;

  return  {
    [=](caf::done_atom, int req_id, const NDArray& result, const RequestStats& stats) {
      if (!self->state.warmup_finished) {
        self->state.num_done_warmups_reqs++;
        if (self->state.num_done_warmups_reqs == self->state.num_warmups_reqs) {
          self->state.warmup_finished = true;
          if (self->state.warmup_waiting) {
            self->state.warmup_rp.deliver(true);
          }
        }

      } else {
        std::cout << "req_id=" << req_id << " elapsed_time_in_micros=" << stats.ElapsedTimeInMicros() << std::endl;
        self->state.num_done_reqs++;
        if (self->state.num_done_reqs == self->state.num_reqs) {
          self->state.finished = true;
          if (self->state.waiting) {
            self->state.rp.deliver(true);
          }
        }
      }
    },
    [=](caf::wait_warmup_atom) {
      self->state.warmup_waiting = true;
      self->state.warmup_rp = self->make_response_promise<bool>();
      assert(!self->state.warmup_finished);
      if (self->state.warmup_finished) {
        self->state.warmup_rp.deliver(true);
      }

      return self->state.warmup_rp;
    },
    [=](caf::wait_atom) {
      self->state.waiting = true;
      self->state.rp = self->make_response_promise<bool>();
      if (self->state.finished) {
        self->state.rp.deliver(true);
      }
 
      return self->state.rp;
    },
  };
}

caf::behavior fin_monitor_fn(caf::stateful_actor<fin_state>* self) {
  return {
    [=](caf::wait_atom) {
      auto rp = self->make_response_promise<bool>();
      self->state.rps.push_back(rp);
      return rp;
    },
    [=](caf::done_atom) {
      for (auto& rp : self->state.rps) {
        rp.deliver(true);
      }

      return true;
    }
  };
}

}
}

