#define CAF_SUITE process_mon
#define FAIL CAF_FAIL

#include <caf/test/dsl.hpp>
#include <dgl/inference/common.h>
#include <dgl/inference/envs.h>

#include "../../src/inference/process/process_control_actor.h"

namespace dgl {
namespace inference{

CAF_TEST_FIXTURE_SCOPE(proc_mon_tests, test_coordinator_fixture<>)


struct state {
  int x;
};

caf::behavior test_fn(caf::stateful_actor<state>* self) {
  return {
    [=](caf::get_atom) {
      std::cerr << "Here???" << std::endl;
      int new_actor_process_global_id = (self->state.x)++;
      std::cerr << "Here???" << std::endl;
      return new_actor_process_global_id;
    }
  };
}

struct cell_state {
  int32_t value = 0;
};

caf::behavior unchecked_cell(caf::stateful_actor<cell_state>* self) {
  return {
    [=](caf::put_atom, int32_t val) { 
      self->state.value = val;
      std::cerr << val << std::endl;
    },
    [=](caf::get_atom) { return self->state.value; },
  };
}

CAF_TEST(process_mon) {
  auto process_mon = sys.spawn(process_monitor_fn);

  auto global_actor_process_id = request<int>(process_mon, caf::get_atom_v);
  CAF_CHECK_EQUAL(global_actor_process_id, 0);

  global_actor_process_id = request<int>(process_mon, caf::get_atom_v);
  CAF_CHECK_EQUAL(global_actor_process_id, 1);
}

CAF_TEST_FIXTURE_SCOPE_END()
}
}
