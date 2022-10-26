#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <dgl/runtime/ndarray.h>

#include <caf/all.hpp>
#include <caf/io/all.hpp>

#include "../common.h"
#include "./custom_types.h"

using NDArray = dgl::runtime::NDArray;

CAF_BEGIN_TYPE_ID_BLOCK(core_extension, first_custom_type_id)

  CAF_ADD_ATOM(core_extension, caf, set_atom, "set")
  // CAF_ADD_ATOM(custom_types, caf, get_atom, "get")
  CAF_ADD_ATOM(core_extension, caf, wait_atom, "wait")
  CAF_ADD_ATOM(core_extension, caf, check_atom, "check")
  CAF_ADD_ATOM(core_extension, caf, initialized_atom, "inited")
  CAF_ADD_ATOM(core_extension, caf, init_atom, "init")
  CAF_ADD_ATOM(core_extension, caf, mpi_broadcast_atom, "mpi_bcast") // mpi broadcast
  CAF_ADD_ATOM(core_extension, caf, mpi_receive_atom, "mpi_rcv") // mpi receive

  // For actor remote lookup
  CAF_ADD_ATOM(core_extension, caf, gloo_ra_atom, "gloo_ra") // gloo_rendezvous_actor
  CAF_ADD_ATOM(core_extension, caf, init_mon_atom, "init_mon") // init_monitor_atom

  CAF_ADD_TYPE_ID(core_extension, (std::vector<char>))
  CAF_ADD_TYPE_ID(core_extension, (std::vector<std::string>))
  CAF_ADD_TYPE_ID(core_extension, (std::chrono::milliseconds))
  CAF_ADD_TYPE_ID(core_extension, (NDArray))

CAF_END_TYPE_ID_BLOCK(core_extension)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(NDArray)
