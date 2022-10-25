#pragma once

#include <unordered_map>
#include <unordered_set>

#include <caf/all.hpp>
#include <caf/io/all.hpp>

#include "./actor/actor_types.h"

namespace dgl {
namespace inference {

class init_monitor_actor : public caf::event_based_actor {
 public:
  init_monitor_actor(caf::actor_config& config, u_int32_t world_size);

 private:
  using actor_names_t = std::vector<std::string>;
  caf::behavior make_behavior() override;

  bool all_initialized(const actor_names_t& actor_names);

  void check_pendings();

  const u_int32_t world_size_;
  std::unordered_map<std::string, std::unordered_set<u_int32_t>> inited_map_;
  std::vector<std::pair<actor_names_t, caf::response_promise>> pendings_;
};

class init_monitor_proxy_actor : public caf::event_based_actor {
 public:
  init_monitor_proxy_actor(caf::actor_config& config,
                           const caf::strong_actor_ptr& global_init_mon_ptr);

 private:
  caf::behavior make_behavior();
  caf::strong_actor_ptr global_init_mon_ptr_;
};

}
}