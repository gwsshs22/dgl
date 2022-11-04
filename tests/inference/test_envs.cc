#define CAF_SUITE envs

#include <caf/test/dsl.hpp>
#include <dgl/inference/common.h>

namespace dgl {
namespace inference{

CAF_TEST(envs) {
  caf::actor_id default_value = 10;
  CAF_CHECK_EQUAL(GetEnv(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, default_value), default_value);

  caf::actor_id changed_value = 2;
  SetEnv(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, changed_value);
  CAF_CHECK_EQUAL(GetEnv(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, default_value), changed_value);

  SetEnv<int>(DGL_INFER_SCHEDULING_TYPE, SchedulingType::kGang);
  CAF_ASSERT(GetEnv<int>(DGL_INFER_SCHEDULING_TYPE, SchedulingType::kIndependent) == SchedulingType::kGang);
  auto t = static_cast<SchedulingType>(GetEnv<int>(DGL_INFER_SCHEDULING_TYPE, SchedulingType::kIndependent));
  CAF_CHECK_EQUAL(t, SchedulingType::kGang);

  SetEnv<bool>(DGL_INFER_USING_AGGREGATION_CACHE, true);
  CAF_CHECK_EQUAL(GetEnv<bool>(DGL_INFER_USING_AGGREGATION_CACHE, false), true);

  SetEnv<int>(DGL_INFER_USING_AGGREGATION_CACHE, 1);
  CAF_CHECK_EQUAL(GetEnv<bool>(DGL_INFER_USING_AGGREGATION_CACHE, false), true);
}

}
}
