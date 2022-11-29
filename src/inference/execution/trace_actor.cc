#include "trace_actor.h"

namespace dgl {
namespace inference {

caf::behavior trace_actor(caf::stateful_actor<trace_actor_state>* self,
                          const std::string& result_dir,
                          int node_rank) {
  self->state.result_dir = result_dir;
  self->state.node_rank = node_rank;
  return {
    [=](caf::put_atom, int batch_id, const std::string& name, int elapsed_micro) {
      if (TRACE_ENABLED) {
        self->state.traces.push_back(std::make_tuple(batch_id, name, elapsed_micro));
      }
    },
    [=](caf::write_trace_atom) {
      std::string file_path = self->state.result_dir + "/node_" + std::to_string(self->state.node_rank) + ".txt";
      remove(file_path.c_str());
      std::fstream fs(file_path, std::fstream::out);
      for (const auto& t : self->state.traces) {
        fs << std::get<0>(t) << "," << std::get<1>(t) << "," << std::get<2>(t) << std::endl;
      }
      fs.close();
      return true;
    }
  };
}

}
}
