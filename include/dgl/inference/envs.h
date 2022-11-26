#pragma once

#include <stdlib.h>
#include <string>

#include <dmlc/parameter.h>
#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

extern const char* DGL_INFER_MASTER_HOST;
extern const char* DGL_INFER_MASTER_PORT;
extern const char* DGL_INFER_NODE_RANK;
extern const char* DGL_INFER_LOCAL_RANK;
extern const char* DGL_INFER_NUM_NODES;
extern const char* DGL_INFER_NUM_BACKUP_SERVERS;
extern const char* DGL_INFER_NUM_DEVICES_PER_NODE;
extern const char* DGL_INFER_IFACE;
extern const char* DGL_INFER_ACTOR_PROCESS_ROLE;
extern const char* DGL_INFER_ACTOR_PROCESS_GLOBAL_ID;

extern const char* DGL_INFER_PARALLELIZATION_TYPE;
extern const char* DGL_INFER_SCHEDULING_TYPE;
extern const char* DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS;

extern const char* DGL_INFER_INPUT_TRACE_DIR;
extern const char* DGL_INFER_NUM_WARMUPS;
extern const char* DGL_INFER_NUM_REQUESTS;
extern const char* DGL_INFER_RESULT_DIR;
extern const char* DGL_INFER_COLLECT_STATS;

enum ParallelizationType {
  kData = 0,
  kP3 = 1,
  kVertexCut = 2,
};

template<typename ValueType>
inline void SetEnv(const char *key,
                   ValueType value) {
  return dmlc::SetEnv(key, value);
}

template<typename ValueType>
inline ValueType GetEnv(const char *key,
                        ValueType default_value) {
  return dmlc::GetEnv(key, default_value);
}

template <typename EnumType>
inline EnumType GetEnumEnv(const char* key) {
  return static_cast<EnumType>(dmlc::GetEnv<int>(key, -1));
}

}
}
