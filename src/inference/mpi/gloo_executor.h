#pragma once

#include <memory>

#include <gloo/rendezvous/store.h>
#include <gloo/rendezvous/context.h>
#include <gloo/transport/tcp/device.h>

#include "../common.h"

namespace dgl {
namespace inference {

class GlooExecutor {
 
 public:
  GlooExecutor(std::unique_ptr<gloo::rendezvous::Store>&& store,
               u_int32_t rank,
               u_int32_t world_size);
 
  void Initialize(const std::string& hostname, const std::string& iface);

  void Broadcast(u_int8_t* ptr, const u_int32_t num_bytes, u_int32_t root, u_int32_t tag);

  void Broadcast(u_int8_t* ptr, const u_int32_t num_bytes, u_int32_t root) {
    Broadcast(ptr, num_bytes, root, 0);
  }

 private:
  std::unique_ptr<gloo::rendezvous::Store> store_;
  std::shared_ptr<gloo::rendezvous::Context> context_;
  std::shared_ptr<gloo::transport::Device> device_;

  const u_int32_t rank_;
  const u_int32_t world_size_;
};

}
}