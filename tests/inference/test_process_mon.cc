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

struct cell_state {
  int32_t value = 0;
};

caf::behavior test_fn(caf::event_based_actor* self) {
  return [](caf::add_atom, int x, const caf::message& m) {
    // std::cout << x << ", " << caf::to_string(m) << std::endl;
    return 1;
  };
}

CAF_TEST(tt) {
  auto ac = sys.spawn(test_fn);
  auto ret = request<int>(ac, caf::add_atom_v, 12, caf::make_message(1,2,3));
  // std::cout << ret << std::endl;
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
