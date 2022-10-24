#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <caf/all.hpp>
#include <caf/io/all.hpp>

#include "./custom_types.h"

CAF_BEGIN_TYPE_ID_BLOCK(core_extension, first_custom_type_id)

  CAF_ADD_ATOM(core_extension, caf, set_atom, "set")
  // CAF_ADD_ATOM(custom_types, caf, get_atom, "get")
  CAF_ADD_ATOM(core_extension, caf, wait_atom, "wait")
  CAF_ADD_ATOM(core_extension, caf, check_atom, "check")
  CAF_ADD_ATOM(core_extension, caf, initialized_atom, "inited")
  CAF_ADD_ATOM(core_extension, caf, init_atom, "init")

  // For actor remote lookup
  CAF_ADD_ATOM(core_extension, caf, gloo_ra_atom, "gloo_ra") // gloo_rendezvous_actor
  CAF_ADD_ATOM(core_extension, caf, mpi_ctr_atom, "mpi_ctr") // mpi_control_actor

  CAF_ADD_TYPE_ID(core_extension, (std::vector<char>))
  CAF_ADD_TYPE_ID(core_extension, (std::vector<std::string>))
  CAF_ADD_TYPE_ID(core_extension, (std::chrono::milliseconds))

  CAF_ADD_TYPE_ID(core_extension, (dgl::inference::MpiInitMsg))

CAF_END_TYPE_ID_BLOCK(core_extension)
