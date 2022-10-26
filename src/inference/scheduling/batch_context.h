#pragma once

#include "../actor/actor_types.h"

namespace dgl {
namespace inference {

struct BatchContext {
  u_int32_t batch_id;
  caf::strong_actor_ptr obj_store_ptr;
};

}
}