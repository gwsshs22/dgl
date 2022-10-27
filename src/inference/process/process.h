/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/inference/exec.h
 * \brief 
 */
#ifndef DGL_INFERENCE_EXEC_H_
#define DGL_INFERENCE_EXEC_H_

#include <caf/all.hpp>
#include <caf/io/all.hpp>

namespace dgl {
namespace inference {


struct config : caf::actor_system_config {
  config() {
    opt_group{custom_options_, "global"}
      .add(port, "port,p", "set port")
      .add(host, "host,H", "set node (ignored in server mode)")
      .add(server_mode, "server-mode,s", "enable server mode");
  }
  uint16_t port = 0;
  std::string host = "localhost";
  bool server_mode = false;
};

void ExecMasterProcess();

void ExecWorkerProcess();

}  // namespace inference
}  // namespace dgl

#endif  // DGL_INFERENCE_EXEC_H_
