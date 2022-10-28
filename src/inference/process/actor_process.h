#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

class Request {

 public:
  Request() = default;
  Request(const Request& other) = default;
  Request(Request&& other) = default;
  Request& operator=(Request&& other) = default;

  Request(const caf::message& message, uint64_t req_id);

  void Done(const caf::message& response);

  inline const caf::message& message() {
    return message_;
  }

 private:
  caf::message message_;
  uint64_t req_id_;
};

void ActorProcessMain(caf::actor_system& system, const config& cfg);

void ActorNotifyInitialized();

Request ActorFetchRequest();

}  // namespace inference
}  // namespace dgl
