#define CAF_SUITE init_mon

#include <atomic>
#include <iostream>
#include <chrono>
#include <thread>

#include <caf/test/dsl.hpp>
#include <dgl/inference/common.h>

#include "../../src/inference/process/init_monitor_actor.h"

namespace dgl {
namespace inference {

CAF_TEST_FIXTURE_SCOPE(init_mon_tests, test_coordinator_fixture<>)

caf::behavior tester_fun(caf::event_based_actor* self, caf::actor init_mon, bool *done) {
  return {
    [=](int) mutable {
      self->request(init_mon, caf::infinite, caf::wait_atom_v, std::vector<std::string>({"service_a", "service_b"}))
          .then(
            [=](bool res) mutable { *done = true; },
            [](caf::error& err){ std::cout << "Error " << caf::to_string(err) << "\n"; });
    }
  };
}

void return_after_sleep_fn(caf::blocking_actor* self,
                           caf::response_promise rp) {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(10ms);
  rp.deliver(true);
}

caf::behavior recv_void_test_fn(caf::event_based_actor* self) {
  return {
    [=](caf::get_atom) {
      auto rp = self->make_response_promise<bool>();
      self->spawn(return_after_sleep_fn, rp);
      return rp;
    }
  };
}

CAF_TEST(init_mon) {
  bool done = false;
  auto init_mon = sys.spawn<init_monitor_actor>();

  anon_send(init_mon, caf::initialized_atom_v, "service_a", 0, 1);
  anon_send(init_mon, caf::initialized_atom_v, "service_b", 0, 2);

  auto tester = sys.spawn(tester_fun, init_mon, &done);
  self->send(tester, 0);
  sched.run();
  CAF_CHECK_EQUAL(done, false);
  anon_send(init_mon, caf::initialized_atom_v, "service_b", 1, 2);
  sched.run();
  CAF_CHECK_EQUAL(done, true);
}

CAF_TEST(recv_void) {
  caf::scoped_actor self { sys };
  auto test_actor = sys.spawn(recv_void_test_fn);

  auto rh = self->request(test_actor, caf::infinite, caf::get_atom_v);
  sched.run();
  receive_result<void>(rh);
}

CAF_TEST_FIXTURE_SCOPE_END()

}
}