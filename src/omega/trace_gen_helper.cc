#include "trace_gen_helper.h"

#include <parallel_hashmap/phmap.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace omega {

typedef phmap::flat_hash_map<int64_t, int64_t> IdMap;

std::vector<IdArray> TraceGenHelper(
    int first_new_gnid,
    const IdArray& infer_target_mask,
    const IdArray& batch_local_ids,
    const IdArray& u,
    const IdArray& v,
    const IdArray& u_in_partitions,
    const IdArray& v_in_partitions,
    const bool independent) {

  int batch_size = batch_local_ids->shape[0];
  int num_edges = u->shape[0];

  IdArray new_gnids = IdArray::Empty({ batch_size }, batch_local_ids->dtype, batch_local_ids->ctx);
  std::vector<int64_t> src_gnids_vec, dst_gnids_vec;

  int64_t* new_gnids_data = static_cast<int64_t*>(new_gnids->data);

  IdMap new_id_map;

  const int64_t* batch_local_ids_data = static_cast<int64_t*>(batch_local_ids->data);
  for (int i = 0; i < batch_size; i++) {
    int new_gnid = first_new_gnid + i;
    *(new_gnids_data + i) = new_gnid;
    new_id_map.insert({*(batch_local_ids_data + i), new_gnid});
  }

  CHECK_EQ(new_id_map.size(), batch_size);

  const int64_t* infer_target_mask_data = static_cast<int64_t*>(infer_target_mask->data);
  const int64_t* u_data = static_cast<int64_t*>(u->data);
  const int64_t* v_data = static_cast<int64_t*>(v->data);
  const int64_t* u_in_partitions_data = static_cast<int64_t*>(u_in_partitions->data);
  const int64_t* v_in_partitions_data = static_cast<int64_t*>(v_in_partitions->data);

  for (int i = 0; i < num_edges; i++) {
    int64_t u_val = *(u_data + i);
    int64_t u_val_in_partitions = *(u_in_partitions_data + i);
    int64_t v_val = *(v_data + i);
    int64_t v_val_in_partitions = *(v_in_partitions_data + i);

    auto u_find_ret = new_id_map.find(u_val);
    auto v_find_ret = new_id_map.find(v_val);

    if (independent) {
      if (infer_target_mask_data[u_val] > 0 && (u_find_ret == new_id_map.end() || u_val != v_val)) {
        continue;
      }
    } else {
      if (infer_target_mask_data[u_val] > 0 && u_find_ret == new_id_map.end()) {
        continue;
      }

      if (infer_target_mask_data[v_val] > 0 && v_find_ret == new_id_map.end()) {
        continue;
      }
    }


    if (u_find_ret == new_id_map.end()) {
      src_gnids_vec.push_back(u_val_in_partitions);
    } else {
      src_gnids_vec.push_back(u_find_ret->second);
    }

    if (v_find_ret == new_id_map.end()) {
      dst_gnids_vec.push_back(v_val_in_partitions);
    } else {
      dst_gnids_vec.push_back(v_find_ret->second);
    }
  }

  return { 
    new_gnids, 
    NDArray::FromVector(src_gnids_vec),
    NDArray::FromVector(dst_gnids_vec)
  };
}

}
}
