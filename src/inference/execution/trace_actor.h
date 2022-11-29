#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

struct trace_actor_state {
  int node_rank;
  std::string result_dir;
  std::vector<std::tuple<int, std::string, int>> traces;
};

caf::behavior trace_actor(caf::stateful_actor<trace_actor_state>* self,
                          const std::string& result_dir,
                          int node_rank);

}
}
