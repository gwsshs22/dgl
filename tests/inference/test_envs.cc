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

  SetEnv<int>(DGL_INFER_PARALLELIZATION_TYPE, ParallelizationType::kVertexCut);
  CAF_ASSERT(GetEnv<int>(DGL_INFER_PARALLELIZATION_TYPE, ParallelizationType::kData) == ParallelizationType::kVertexCut);
  auto t = static_cast<ParallelizationType>(GetEnv<int>(DGL_INFER_PARALLELIZATION_TYPE, ParallelizationType::kData));
  CAF_CHECK_EQUAL(t, ParallelizationType::kVertexCut);

  SetEnv<bool>(DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS, true);
  CAF_CHECK_EQUAL(GetEnv<bool>(DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS, false), true);

  SetEnv<int>(DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS, 1);
  CAF_CHECK_EQUAL(GetEnv<bool>(DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS, false), true);
}

CAF_TEST(hash) {
  constexpr int N = 4;
  int counter[N] = {0, };

  for (uint64_t i = 0; i < 1000000; i++) {
    counter[hash(i) % N]++;
  }

  for (int i = 0; i < N; i++) {
    std::cout << "[" << i << "] = " << counter[i] << ", ";
  }
  std::cout << std::endl;
}

}
}
