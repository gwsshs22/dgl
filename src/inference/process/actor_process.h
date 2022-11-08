#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

struct ActorRequest {

  ActorRequest() = default;
  ActorRequest(const ActorRequest& other) = default;
  ActorRequest(ActorRequest&& other) = default;
  ActorRequest& operator=(ActorRequest&& other) = default;
  ActorRequest(uint64_t req_id, int request_type, int batch_id);

  inline uint64_t req_id() {
    return req_id_;
  }

  inline int request_type() {
    return request_type_;
  }

  inline int batch_id() {
    return batch_id_;
  }

 private:
  uint64_t req_id_;
  int request_type_;
  int batch_id_;
};

void ActorProcessMain(caf::actor_system& system, const config& cfg);

void ActorNotifyInitialized();

ActorRequest ActorFetchRequest();

void ActorRequestDone(uint64_t req_id);

}  // namespace inference
}  // namespace dgl
