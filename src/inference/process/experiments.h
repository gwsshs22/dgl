#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

void input_feader_fn(caf::blocking_actor* self,
                     const caf::actor& scheduler,
                     const std::string& input_trace_dir,
                     int num_warmup_reqs,
                     int num_reqs);

enum receiver_status {
  k
};

struct result_receiver_state {
  caf::response_promise warmup_rp;
  caf::response_promise rp;
  int num_warmups_reqs;
  int num_reqs;
  int num_done_warmups_reqs;
  int num_done_reqs;
  bool warmup_finished;
  bool warmup_waiting;
  bool finished;
  bool waiting;
};

caf::behavior result_receiver_fn(caf::stateful_actor<result_receiver_state>* self,
                                 int num_warmup_reqs,
                                 int num_reqs);

struct fin_state {
  std::vector<caf::response_promise> rps;
};

caf::behavior fin_monitor_fn(caf::stateful_actor<fin_state>* self);

}
}
