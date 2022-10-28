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
extern const char* DGL_INFER_NUM_DEVICES_PER_NODE;
extern const char* DGL_INFER_IFACE;
extern const char* DGL_INFER_ACTOR_PROCESS_ROLE;
extern const char* DGL_INFER_ACTOR_PROCESS_GLOBAL_ID;

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

}
}
