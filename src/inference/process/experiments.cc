#include "experiments.h"

#include <fstream>

namespace dgl {
namespace inference {

constexpr uint64_t kDGLSerialize_Tensors = 0xDD5A9FBE3FA2443F;
typedef std::pair<std::string, NDArray> NamedTensor;

std::map<std::string, NDArray> read_tensor_dict(const std::string& filename) {
    auto fs = std::unique_ptr<dmlc::Stream>(
      dmlc::Stream::Create(filename.c_str(), "r"));
    CHECK(fs) << "Filename is invalid or file doesn't exists";
    uint64_t magincNum, num_elements;
    CHECK(fs->Read(&magincNum)) << "Invalid file";
    CHECK_EQ(magincNum, kDGLSerialize_Tensors) << "Invalid DGL tensor file";
    CHECK(fs->Read(&num_elements)) << "Invalid num of elements";
    std::map<std::string, NDArray> nd_dict;
    std::vector<NamedTensor> namedTensors;
    fs->Read(&namedTensors);

    for (auto kv : namedTensors) {
      nd_dict[kv.first] = kv.second;
    }
    
    return nd_dict;
}

int get_num_traces(const std::string& input_trace_dir) {
  int num_traces;
  std::ifstream infile(input_trace_dir + "/num_traces.txt");
  infile >> num_traces;
  return num_traces;
}

void input_feader_fn(caf::blocking_actor* self,
                     const caf::actor& scheduler,
                     const std::string& input_trace_dir,
                     int num_warmup_reqs,
                     int num_reqs) {

  auto num_traces = get_num_traces(input_trace_dir);

  auto new_gnids_vec = std::vector<NDArray>();
  auto new_features_vec = std::vector<NDArray>();
  auto src_gnids_vec = std::vector<NDArray>();
  auto dst_gnids_vec = std::vector<NDArray>();

  std::cout << "input_feader start loading (" << num_traces << ") input traces" << std::endl;
  for (int i = 0; i < num_traces; i++) {
    auto nd_dict = read_tensor_dict(input_trace_dir + "/" + std::to_string(i) + ".dgl");
    new_gnids_vec.push_back(nd_dict["new_gnids"]);
    new_features_vec.push_back(nd_dict["new_features"]);
    src_gnids_vec.push_back(nd_dict["src_gnids"]);
    dst_gnids_vec.push_back(nd_dict["dst_gnids"]);
  }
  

  for (int i = 0; i < num_warmup_reqs; i++) {
    auto idx = i % num_traces;
    auto rh = self->request(scheduler, caf::infinite, caf::enqueue_atom_v, new_gnids_vec[idx], new_features_vec[idx], src_gnids_vec[idx], dst_gnids_vec[idx], std::chrono::steady_clock::now());
    receive_result<int>(rh);
  }

  // Wait for warmup done.
  std::cout << "input_feader wait start message" << std::endl;
  self->receive([](caf::start_atom){});

  for (int i = 0; i < num_reqs; i++) {
    auto idx = i % num_traces;
    auto rh = self->request(scheduler, caf::infinite, caf::enqueue_atom_v, new_gnids_vec[idx], new_features_vec[idx], src_gnids_vec[idx], dst_gnids_vec[idx], std::chrono::steady_clock::now());
    receive_result<int>(rh);
  }

  // Wait for all done
  self->receive([](caf::done_atom){});
}

caf::behavior result_receiver_fn(caf::stateful_actor<result_receiver_state>* self,
                                 const caf::actor& trace_actor,
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
        auto elapsed_time_in_micros = stats.ElapsedTimeInMicros();
        std::cout << "[WARMUP] req_id=" << req_id << " elapsed_time_in_micros=" << stats.ElapsedTimeInMicros() << ", result=" << result << std::endl;
        self->send(trace_actor, caf::put_atom_v, req_id, "total", elapsed_time_in_micros);

        self->state.num_done_warmups_reqs++;
        if (self->state.num_done_warmups_reqs == self->state.num_warmups_reqs) {
          self->state.warmup_finished = true;
          if (self->state.warmup_waiting) {
            self->state.warmup_rp.deliver(true);
          }
        }
      } else {
        auto elapsed_time_in_micros = stats.ElapsedTimeInMicros();
        caf::aout(self) << "req_id=" << req_id << " elapsed_time_in_micros=" << elapsed_time_in_micros << std::endl;
        self->send(trace_actor, caf::put_atom_v, req_id, "total", elapsed_time_in_micros);

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

