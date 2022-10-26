#pragma once

#include <caf/all.hpp>
#include <caf/io/all.hpp>

#include "./actor/actor_types.h"

namespace dgl {
namespace inference {

class obj_store_actor : public caf::event_based_actor {

 public:
  obj_store_actor(caf::actor_config& config);

 private:
  caf::behavior make_behavior() override;

};

}
}
