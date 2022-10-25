#pragma once

#include <unordered_map>

#include <caf/all.hpp>
#include <caf/io/all.hpp>

#include <gloo/common/error.h>
#include <gloo/rendezvous/store.h>

#include "actor/actor_types.h"

namespace dgl {
namespace inference {

using gloo_rendezvous_actor_type
  = caf::typed_actor<caf::reacts_to<caf::set_atom, std::string, std::vector<char>>,
                     caf::replies_to<caf::get_atom, std::string>::with<std::vector<char>>,
                     caf::replies_to<caf::wait_atom, std::vector<std::string>, u_int64_t>::with<bool>,
                     caf::reacts_to<caf::check_atom>>;

class gloo_rendezvous_actor : public gloo_rendezvous_actor_type::base {

 public:
  gloo_rendezvous_actor(caf::actor_config& config);

 protected:
  gloo_rendezvous_actor_type::behavior_type make_behavior() override;
 private:
  bool ContainsAll(std::vector<std::string> keys);
  void CheckPendings();

  std::unordered_map<std::string, std::vector<char>> store_;
  std::vector<std::tuple<caf::response_promise,
                         std::vector<std::string>,
                         u_int64_t,
                         std::chrono::steady_clock::time_point>> pendings_;
};

class ActorBasedStore : public gloo::rendezvous::Store {
 public:
  ActorBasedStore(caf::actor_system& actor_system,
                  const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr);

  virtual ~ActorBasedStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(const std::vector<std::string>& keys) override {
    wait(keys, Store::kDefaultTimeout);
  }

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

 private:
  std::unique_ptr<caf::scoped_actor> self_;
  caf::actor gloo_rendezvous_actor_;
};

}  // namespace inference
}  // namespace dgl
