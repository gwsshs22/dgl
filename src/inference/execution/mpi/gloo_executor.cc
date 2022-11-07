#include "gloo_executor.h"

#include <gloo/broadcast.h>

#include <iostream>

namespace dgl {
namespace inference {

GlooExecutor::GlooExecutor(std::unique_ptr<gloo::rendezvous::Store>&& store,
                           u_int32_t rank,
                           u_int32_t num_nodes)
    : store_(std::move(store)), rank_(rank), num_nodes_(num_nodes) {
}

void GlooExecutor::Initialize(const std::string& hostname, const std::string& iface) {
  gloo::transport::tcp::attr attr;
  attr.hostname = hostname;
  attr.iface = iface;
  attr.ai_family = AF_UNSPEC;

  context_ = std::make_shared<gloo::rendezvous::Context>(rank_, num_nodes_);
  device_ = gloo::transport::tcp::CreateDevice(attr);  
  
  context_->connectFullMesh(*store_, device_);
}

void GlooExecutor::Broadcast(u_int8_t* ptr, const u_int32_t num_bytes, u_int32_t root, u_int32_t tag) {
  auto broadcast_opts = gloo::BroadcastOptions(context_);
  broadcast_opts.setOutput(ptr, num_bytes);
  broadcast_opts.setRoot(root);
  broadcast_opts.setTag(tag);
  gloo::broadcast(broadcast_opts);
}

void GlooExecutor::Send(u_int8_t* ptr, const u_int32_t num_bytes, u_int32_t dst_rank, u_int32_t tag) {
  auto ubuf = context_->createUnboundBuffer(reinterpret_cast<void*>(ptr), num_bytes);
  ubuf->send(dst_rank, tag);
  ubuf->waitSend();
}

void GlooExecutor::Recv(u_int8_t* ptr, const u_int32_t num_bytes, u_int32_t src_rank, u_int32_t tag) {
  auto ubuf = context_->createUnboundBuffer(reinterpret_cast<void*>(ptr), num_bytes);
  ubuf->recv(src_rank, tag);
  ubuf->waitRecv();
}

}
}
