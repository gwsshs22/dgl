#include "actor_process.h"

#include <thread>

#include <dmlc/blockingconcurrentqueue.h>

#include <dgl/runtime/semaphore_wrapper.h>

#include "process_control_actor.h"

namespace dgl {
namespace inference {

static dgl::runtime::Semaphore sem_;
static dmlc::moodycamel::BlockingConcurrentQueue<Request> queue_;
static caf::actor process_request_handler_;

Request::Request(const caf::message& message, uint64_t req_id)
    : message_(message), req_id_(req_id) {
}

void Request::Done(const caf::message& response) {
  anon_send(process_request_handler_, caf::response_atom_v, req_id_, response);
}

struct process_request_handler_state {
  bool initialized = false;
  caf::actor requester_actor;
};

caf::behavior process_request_handler_fn(caf::stateful_actor<process_request_handler_state>* self,
                                         const caf::strong_actor_ptr& process_mon_actor_ptr,
                                         int actor_process_global_id) {
  return {
    [=](caf::connect_atom) {
      self->state.requester_actor = caf::actor_cast<caf::actor>(self->current_sender());
    },
    [=](caf::request_atom, uint64_t req_id, const caf::message& message) {
      queue_.enqueue(Request(message, req_id));
    },
    [=](caf::response_atom, uint64_t req_id, const caf::message& response) {
      self->send(self->state.requester_actor, caf::response_atom_v, req_id, response);
    },
    [=](caf::initialized_atom) {
      if (self->state.initialized) {
        return;
      }

      self->state.initialized = true;
      auto process_monitor = caf::actor_cast<caf::actor>(process_mon_actor_ptr);
      self->send(process_monitor, caf::initialized_atom_v, actor_process_global_id);
    }
  };
}

void ActorProcessMain(caf::actor_system& system, const config& cfg) {
  auto master_host = GetEnv<std::string>(DGL_INFER_MASTER_HOST, "localhost");
  auto master_port = GetEnv<u_int16_t>(DGL_INFER_MASTER_PORT, 0);
  auto actor_process_global_id = GetEnv<int>(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, -1);
  auto role = GetEnv<std::string>(DGL_INFER_ACTOR_PROCESS_ROLE, "");
  auto local_rank = GetEnv<int>(DGL_INFER_LOCAL_RANK, -1);
  // TODO: env variables validation

  auto node = retry<caf::node_id>([&] {
    return system.middleman().connect(master_host, master_port);
  });

  if (!node) {
    // TODO: error handling
    std::cerr << "*** connect failed: " << caf::to_string(node.error()) << std::endl;
    return;
  }

  auto process_mon_actor_ptr = system.middleman().remote_lookup(caf::process_mon_atom_v, node.value());
  process_request_handler_ = system.spawn(process_request_handler_fn,
      process_mon_actor_ptr,
      actor_process_global_id);

  sem_.Post();

  system.await_all_actors_done();
}

void ActorNotifyInitialized() {
  sem_.Wait();
  anon_send(process_request_handler_, caf::initialized_atom_v);
}

Request ActorFetchRequest() {
  Request req;
  queue_.wait_dequeue<Request>(req);
  return req;
}

}
}
