#include "gloo_rendezvous_actor.h"

#include <iostream>

namespace dgl {
namespace inference {


gloo_rendezvous_actor::gloo_rendezvous_actor(caf::actor_config& config)
    : gloo_rendezvous_actor_type::base(config) {
}

bool gloo_rendezvous_actor::ContainsAll(std::vector<std::string> keys) {
  auto done = true;
  for (const auto& key : keys) {
    if (store_.find(key) == store_.end()) {
      done = false;
      break;
    }
  }

  return done;
}

void gloo_rendezvous_actor::CheckPendings() {
  auto it = pendings_.begin();
  while (it != pendings_.end()) {
    auto& t = *it;
    auto& promise = std::get<0>(t);
    auto& keys = std::get<1>(t);
    auto& timeout = std::get<2>(t);
    auto& request_arrival_time = std::get<3>(t);

    if (ContainsAll(keys)) {
      promise.deliver(true);
      it = pendings_.erase(it);
    } else {
      auto now = std::chrono::steady_clock::now();
      if (request_arrival_time + std::chrono::milliseconds(timeout) <= now) {
        promise.deliver(caf::make_error(caf::sec::request_timeout, "Gloo wait timeout"));
        GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG(
            "Wait timeout for key(s): ", ::gloo::MakeString(keys)));
      } else {
        ++it;
      }
    }
  }
}

gloo_rendezvous_actor_type::behavior_type gloo_rendezvous_actor::make_behavior() {
  return {
    [=](caf::set_atom, const std::string& key, const std::vector<char>& data) {
      store_[key] = data;
      CheckPendings();
    },
    [=](caf::get_atom, const std::string& key) {
      auto it = store_.find(key);
      if (it == store_.end()) {
        return std::vector<char>();
      }

      return it->second;
    },
    [=](caf::wait_atom, const std::vector<std::string>& keys, u_int64_t timeout) {
      auto now = std::chrono::steady_clock::now();
      auto rp = this->make_response_promise<bool>();
      if (ContainsAll(keys)) {
        rp.deliver(true);
      } else {
        pendings_.emplace_back(std::make_tuple(rp, keys, timeout, now));
        this->delayed_send(this, std::chrono::milliseconds(timeout + 10), caf::check_atom_v);
      }

      return rp;
    },
    [=](caf::check_atom) {
      CheckPendings();
    }
  };
}

ActorBasedStore::ActorBasedStore(caf::actor_system& actor_system,
                                 const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr)
    : gloo_rendezvous_actor_(caf::actor_cast<caf::actor>(gloo_rendezvous_actor_ptr)) {
  self_ = std::make_unique<caf::scoped_actor>(actor_system);
}

void ActorBasedStore::set(const std::string& key, const std::vector<char>& data) {
  (*self_)->send(gloo_rendezvous_actor_, caf::set_atom_v, key, data);
}

std::vector<char> ActorBasedStore::get(const std::string& key) {
  std::vector<char> res;
  (*self_)->request(gloo_rendezvous_actor_, caf::infinite, caf::get_atom_v, key).receive(
    [&](caf::error& x) {
      GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG(
        "Cannot get the result for key: ", ::gloo::MakeString(key)));
    },
    [&](std::vector<char> reply) {
      res = reply;
    }
  );

  return res;
}

void ActorBasedStore::wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) {
  u_int64_t timeout_millis = timeout.count();
  (*self_)->request(gloo_rendezvous_actor_, caf::infinite, caf::wait_atom_v, keys, timeout_millis).receive(
    [&](caf::error& x) {
      GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG(
        "Cannot get the result for key: ", ::gloo::MakeString(keys)));
    },
    [&](bool res) {
    }
  );
}

}
}
