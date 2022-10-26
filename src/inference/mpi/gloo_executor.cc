#include "gloo_executor.h"

#include <gloo/broadcast.h>

#include <iostream>

namespace dgl {
namespace inference {

GlooExecutor::GlooExecutor(std::unique_ptr<gloo::rendezvous::Store>&& store,
                           u_int32_t rank,
                           u_int32_t world_size)
    : store_(std::move(store)), rank_(rank), world_size_(world_size) {
}

void GlooExecutor::Initialize(const std::string& hostname, const std::string& iface) {
  gloo::transport::tcp::attr attr;
  attr.hostname = hostname;
  attr.iface = iface;
  attr.ai_family = AF_UNSPEC;

  context_ = std::make_shared<gloo::rendezvous::Context>(rank_, world_size_);
  device_ = gloo::transport::tcp::CreateDevice(attr);  
  
  context_->connectFullMesh(*store_, device_);
}

void GlooExecutor::Broadcast(u_int8_t* ptr, const u_int32_t num_bytes, u_int32_t root, u_int32_t tag = 0) {
  auto broadcast_opts = gloo::BroadcastOptions(context_);
  broadcast_opts.setOutput(ptr, num_bytes);
  broadcast_opts.setRoot(root);
  broadcast_opts.setTag(tag);
  gloo::broadcast(broadcast_opts);
}

}
}