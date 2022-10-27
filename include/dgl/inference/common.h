#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <iostream>

#include <caf/all.hpp>
#include <caf/io/all.hpp>

#include <dgl/inference/types.h>
#include <dgl/runtime/ndarray.h>

CAF_BEGIN_TYPE_ID_BLOCK(core_extension, first_custom_type_id)

  CAF_ADD_ATOM(core_extension, caf, set_atom, "set")
  CAF_ADD_ATOM(core_extension, caf, wait_atom, "wait")
  CAF_ADD_ATOM(core_extension, caf, check_atom, "check")
  CAF_ADD_ATOM(core_extension, caf, initialized_atom, "inited")
  CAF_ADD_ATOM(core_extension, caf, init_atom, "init")
  CAF_ADD_ATOM(core_extension, caf, mpi_broadcast_atom, "mpi_bcast") // mpi broadcast
  CAF_ADD_ATOM(core_extension, caf, mpi_receive_atom, "mpi_rcv") // mpi receive
  CAF_ADD_ATOM(core_extension, caf, enqueue_atom, "enqueue") // enqueue
  CAF_ADD_ATOM(core_extension, caf, exec_atom, "exec") // exec
  CAF_ADD_ATOM(core_extension, caf, done_atom, "done") // done

  // For actor remote lookup
  CAF_ADD_ATOM(core_extension, caf, gloo_ra_atom, "gloo_ra") // gloo_rendezvous_actor
  CAF_ADD_ATOM(core_extension, caf, init_mon_atom, "init_mon") // init_monitor_actor
  CAF_ADD_ATOM(core_extension, caf, exec_control_atom, "exec_ctl") // executor_control_actor

  CAF_ADD_TYPE_ID(core_extension, (dgl::runtime::NDArray))
  CAF_ADD_TYPE_ID(core_extension, (dgl::inference::TaskType))
  

CAF_END_TYPE_ID_BLOCK(core_extension)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(dgl::runtime::NDArray)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<dgl::runtime::NDArray>)

namespace dgl {
namespace inference {

template <typename Actor>
void ReportToInitMon(Actor& self, std::string actor_name, int rank, int world_size) {
  auto init_mon_ptr = self.system().registry().get(caf::init_mon_atom_v);
  auto init_mon = caf::actor_cast<caf::actor>(init_mon_ptr);
  self.send(init_mon, caf::initialized_atom_v, actor_name, rank, world_size);
}

template <typename T>
inline T receive_result(caf::response_handle<caf::blocking_actor, caf::message, true>& hdl) {
  T hold;
  hdl.receive(
    [&](const T& res) { hold = std::move(res); },
    [&](caf::error& err) {
      // TODO: error handling.
      caf::aout(hdl.self()) << "Error : " << caf::to_string(err) << std::endl;
    }
  );

  return hold;
}

template <>
inline void receive_result(caf::response_handle<caf::blocking_actor, caf::message, true>& hdl) {
  hdl.receive(
    [&]() { },
    [&](caf::error& err) {
      // TODO: error handling.
      caf::aout(hdl.self()) << "Error : " << caf::to_string(err) << std::endl;
    }
  );

  return;
}

}
}
