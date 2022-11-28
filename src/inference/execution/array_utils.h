#pragma once

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

inline NDArray _SplitFeatureForNodeByDimensonZero(const NDArray& new_features, int start_idx, int end_idx) {
  auto bytes_per_elem = (new_features->dtype.bits * new_features->dtype.lanes + 7) / 8;
  auto bytes_per_row = bytes_per_elem * new_features->shape[1];
  auto length = end_idx - start_idx;

  NDArray ret = NDArray::Empty({ length, new_features->shape[1] }, new_features->dtype, new_features->ctx);

  auto src_offset = bytes_per_row * start_idx;
  auto copy_bytes = bytes_per_row * (end_idx - start_idx);
  memcpy(ret->data, ((char*) new_features->data) + src_offset, copy_bytes);
  return ret;
}

inline NDArray _SplitFeatureForNodeByDimensonOne(const NDArray& new_features, int start_idx, int end_idx) {
  auto bytes_per_elem = (new_features->dtype.bits * new_features->dtype.lanes + 7) / 8;
  auto bytes_per_row = bytes_per_elem * new_features->shape[1];
  auto length = end_idx - start_idx;
  
  auto bytes_per_copied_row = bytes_per_elem * length;
  auto batch_size = new_features->shape[0];

  NDArray ret = NDArray::Empty({ batch_size, length }, new_features->dtype, new_features->ctx);

  for (int i = 0; i < batch_size; i++) {
    auto src_offset = i * bytes_per_row + start_idx * bytes_per_elem;
    auto dst_offset = i * bytes_per_copied_row;
    auto copy_bytes = bytes_per_copied_row;
    memcpy(((char*) ret->data) + dst_offset, ((char*) new_features->data) + src_offset, copy_bytes);
  }

  return ret;
}

inline NDArray SplitFeatureForNode(const NDArray& new_features, const FeatureSplitMethod& split_method, int node_rank) {
  CHECK_EQ(new_features->ndim, 2);
  int start_idx = split_method.split[node_rank];
  int end_idx = split_method.split[node_rank + 1];
  if (split_method.split_dimension == 0) {
    return _SplitFeatureForNodeByDimensonZero(new_features, start_idx, end_idx);
  } else {
    CHECK_EQ(split_method.split_dimension, 1);
    return _SplitFeatureForNodeByDimensonOne(new_features, start_idx, end_idx);
  }
}

inline std::vector<int> _GetSplit(int num_split, int size) {
  auto s = std::vector<int>(num_split + 1);
  s[0] = 0;
  for (int i = 0; i < num_split; i++) {
    s[i + 1] = size / num_split;
  }

  for (int i = 0; i < size % num_split; i++) {
    s[i + 1]++;
  }

  for (int i = 1; i <= num_split; i++) {
    s[i] += s[i-1];
  }

  return s;
}

inline FeatureSplitMethod GetP3FeatureSplit(int num_nodes, int batch_size, int feature_size) {
  FeatureSplitMethod split_method;
  split_method.split_dimension = 1;
  split_method.split = _GetSplit(num_nodes, feature_size);
  return split_method;
}

inline FeatureSplitMethod GetVcutFeatureSplit(int num_nodes, int batch_size, int feature_size) {
  FeatureSplitMethod split_method;
  split_method.split_dimension = 0;
  split_method.split = _GetSplit(num_nodes, batch_size);
  return split_method;
}

}
}
