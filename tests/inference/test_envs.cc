#define CAF_SUITE envs

#include <caf/test/dsl.hpp>
#include <dgl/inference/common.h>
#include <dgl/inference/envs.h>

namespace dgl {
namespace inference{

CAF_TEST(envs) {
  caf::actor_id default_value = 10;
  CAF_CHECK_EQUAL(GetEnv(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, default_value), default_value);

  caf::actor_id changed_value = 2;
  SetEnv(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, changed_value);
  CAF_CHECK_EQUAL(GetEnv(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, default_value), changed_value);
}

}
}
