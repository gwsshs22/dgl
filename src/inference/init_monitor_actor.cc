#include "init_monitor_actor.h"

namespace dgl {
namespace inference {

init_monitor_actor::init_monitor_actor(caf::actor_config& config)
    : event_based_actor(config) {
}

caf::behavior init_monitor_actor::make_behavior() {
  return {
    [&](caf::initialized_atom, const std::string& actor_name, int rank, int world_size) {
      assert(world_size > 0);
      assert(rank < world_size);
      assert(rank >= 0);
      
      auto it = inited_map_.find(actor_name);
      if (it == inited_map_.end()) {
        auto rank_set = std::unordered_set<u_int32_t>();
        rank_set.insert(rank);
        inited_map_[actor_name] = std::move(rank_set);
        world_size_map_[actor_name] = world_size;
      } else {
        auto& rank_set = it->second;
        assert(rank_set.find(rank) == rank_set.end());
        assert(world_size == world_size_map_[actor_name]);

        rank_set.insert(rank);
      }

      check_pendings();
    },
    [&](caf::wait_atom, const actor_names_t& actor_names) {
      auto rp = make_response_promise<bool>();
      if (all_initialized(actor_names)) {
        rp.deliver(true);
      } else {
        pendings_.emplace_back(std::make_pair(actor_names, rp));
      }

      return rp;
    }
  };
}

bool init_monitor_actor::all_initialized(const actor_names_t& actor_names) {
  for (const auto& name : actor_names) {
    auto it = inited_map_.find(name);
    if (it == inited_map_.end()) {
      return false;
    }
    
    auto inited_ranks = it->second;
    if (inited_ranks.size() < world_size_map_[name]) {
      return false;
    }
  }

  return true;
}

void init_monitor_actor::check_pendings() {
  auto it = pendings_.begin();
  while (it != pendings_.end()) {
    auto actor_names = it->first;
    if (all_initialized(actor_names)) {
      it->second.deliver(true);
      it = pendings_.erase(it);
    } else {
      it++;
    }
  }
}

init_monitor_proxy_actor::init_monitor_proxy_actor(caf::actor_config& config,
                                                   const caf::strong_actor_ptr& global_init_mon_ptr)
    : event_based_actor(config), global_init_mon_ptr_(global_init_mon_ptr) {
}

caf::behavior init_monitor_proxy_actor::make_behavior() {
  caf::actor global_init_mon = caf::actor_cast<caf::actor>(global_init_mon_ptr_);
  return {
    [=](caf::initialized_atom, const std::string& actor_name, int rank, int world_size) {
      send(global_init_mon, caf::initialized_atom_v, actor_name, rank, world_size);
    },
    [=](caf::wait_atom, const std::vector<std::string>& actor_names) {
      auto rp = make_response_promise<bool>();
      request(global_init_mon, caf::infinite, caf::wait_atom_v, actor_names).then(
        [=]() mutable {
          rp.deliver(true);
        });
      return rp;
    }
  };
}

}
}
