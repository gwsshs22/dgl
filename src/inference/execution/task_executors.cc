#include "task_executors.h"
#include "mem_utils.h"
#include "array_utils.h"

namespace dgl {
namespace inference {

inline void __move_input_to_shared_mem_one_tensor_body(caf::blocking_actor *self,
                                                       const caf::actor& object_storage_actor,
                                                       const std::string& name,
                                                       const NDArray& arr) {
  auto rh = self->request(object_storage_actor, caf::infinite, caf::put_atom_v, name, arr);
  receive_result<bool>(rh);
  rh = self->request(object_storage_actor, caf::infinite, caf::move_to_shared_atom_v, name);
  receive_result<bool>(rh);
}

void __recv_and_move_one_tensor_body(caf::blocking_actor* self,
                                     const caf::actor& mpi_actor,
                                     const caf::actor& object_storage_actor,
                                     int batch_id,
                                     const std::string name,
                                     const uint32_t tag,
                                     bool is_brecv) {
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;
  if (is_brecv) {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, 0, tag);
    rh_list.push_back(rh);
  } else {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, 0, tag);
    rh_list.push_back(rh);
  }

  auto arr = receive_result<NDArray>(rh_list[0]);
  rh_list.clear();

  __move_input_to_shared_mem_one_tensor_body(self, object_storage_actor, name, arr);
}

void _recv_and_move_one_tensor(caf::blocking_actor* self,
                               const caf::actor& mpi_actor,
                               const caf::actor& object_storage_actor,
                               int batch_id,
                               const std::string name,
                               const uint32_t tag,
                               bool is_brecv) {
  __recv_and_move_one_tensor_body(self, mpi_actor, object_storage_actor, batch_id, name, tag, is_brecv);
  self->receive([](caf::get_atom) { return true; });
}

void _move_input_to_shared_mem_one_tensor(caf::blocking_actor *self,
                                          const caf::actor& object_storage_actor,
                                          const std::string& name,
                                          const NDArray& arr) {
  __move_input_to_shared_mem_one_tensor_body(self, object_storage_actor, name, arr);
  self->receive([](caf::get_atom) { return true; });
}

void input_bsend_all_fn(caf::blocking_actor* self,
                        const caf::actor& mpi_actor,
                        const caf::actor& trace_actor,
                        const caf::actor& object_storage_actor,
                        int batch_id,
                        const NDArray& new_gnids,
                        const NDArray& new_features,
                        const NDArray& src_gnids,
                        const NDArray& dst_gnids,
                        const uint32_t tag) {
  // TraceMe send(batch_id, "input_send");
  auto start_time = std::chrono::steady_clock::now();
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, new_gnids, tag));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, src_gnids, tag + 1));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, dst_gnids, tag + 2));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, new_features, tag + 3));

  auto move_fn = [&](const std::string name, const NDArray& arr) {
    auto actor = self->spawn(_move_input_to_shared_mem_one_tensor, object_storage_actor, name, arr);
    return self->request(actor, caf::infinite, caf::get_atom_v);
  };

  rh_list.push_back(move_fn("new_gnids", new_gnids));
  rh_list.push_back(move_fn("src_gnids", src_gnids));
  rh_list.push_back(move_fn("dst_gnids", dst_gnids));
  rh_list.push_back(move_fn("new_features", new_features));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  if (TRACE_ENABLED) {
    int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
    self->send(trace_actor, caf::put_atom_v, batch_id, "input_send", elapsed);
  }
  self->receive([](caf::get_atom) { });
}

void input_brecv_all_fn(caf::blocking_actor* self,
                        const caf::actor& mpi_actor,
                        const caf::actor& object_storage_actor,
                        int batch_id,
                        const uint32_t tag) {
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  auto recv_fn = [&](const std::string name, int tag) {
    auto actor = self->spawn(_recv_and_move_one_tensor,
                             mpi_actor,
                             object_storage_actor,
                             batch_id,
                             name,
                             tag,
                             true);
    return self->request(actor, caf::infinite, caf::get_atom_v);
  };

  rh_list.push_back(recv_fn("new_gnids", tag));
  rh_list.push_back(recv_fn("src_gnids", tag + 1));
  rh_list.push_back(recv_fn("dst_gnids", tag + 2));
  rh_list.push_back(recv_fn("new_features", tag + 3));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  self->receive([](caf::get_atom) {});
}

void _send_feature_split(caf::blocking_actor* self,
                         const caf::actor& mpi_actor,
                         int node_rank,
                         const NDArray& new_features,
                         const FeatureSplitMethod& split_method,
                         uint32_t tag) {
  auto feature_split = SplitFeatureForNode(new_features, split_method, node_rank);
  auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, feature_split, tag);
  receive_result<bool>(rh);
  self->receive([](caf::get_atom) { return true; });
}

void _move_feature_split(caf::blocking_actor* self,
                         const caf::actor& object_storage_actor,
                         const NDArray& new_features,
                         const FeatureSplitMethod& split_method) {
  auto feature_split = SplitFeatureForNode(new_features, split_method, 0);
  __move_input_to_shared_mem_one_tensor_body(self, object_storage_actor, "new_features", feature_split);
  self->receive([](caf::get_atom) { return true; });
}

void input_bsend_scatter_fn(caf::blocking_actor* self,
                            const caf::actor& mpi_actor,
                            const caf::actor& trace_actor,
                            const caf::actor& object_storage_actor,
                            int num_nodes,
                            BroadcastInitType init_type,
                            int batch_id,
                            const NDArray& new_gnids,
                            const NDArray& new_features,
                            const NDArray& src_gnids,
                            const NDArray& dst_gnids,
                            const FeatureSplitMethod& split_method,
                            const uint32_t tag) {
  // TraceMe send(batch_id, "input_send");
  auto start_time = std::chrono::steady_clock::now();
  assert(init_type == BroadcastInitType::kScatter || init_type == BroadcastInitType::kScatterFeatureOnly);

  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  if (init_type == BroadcastInitType::kScatter) {
    rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, new_gnids, tag));
    rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, src_gnids, tag + 1));
    rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, dst_gnids, tag + 2));
  }

  for (int i = 1; i < num_nodes; i++) {
    auto split_sender = self->spawn(_send_feature_split, mpi_actor, i, new_features, split_method, tag + 3 + i);
    rh_list.push_back(self->request(split_sender, caf::infinite, caf::get_atom_v));
  }

  auto move_fn = [&](const std::string name, const NDArray& arr) {
    auto actor = self->spawn(_move_input_to_shared_mem_one_tensor, object_storage_actor, name, arr);
    return self->request(actor, caf::infinite, caf::get_atom_v);
  };

  rh_list.push_back(move_fn("new_gnids", new_gnids));
  rh_list.push_back(move_fn("src_gnids", src_gnids));
  rh_list.push_back(move_fn("dst_gnids", dst_gnids));
  auto split_mover = self->spawn(_move_feature_split, object_storage_actor, new_features, split_method);
  rh_list.push_back(self->request(split_mover, caf::infinite, caf::get_atom_v));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  if (TRACE_ENABLED) {
    int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
    self->send(trace_actor, caf::put_atom_v, batch_id, "input_send", elapsed);
  }

  self->receive([](caf::get_atom) { });
}

void input_brecv_scatter_fn(caf::blocking_actor* self,
                            const caf::actor& mpi_actor,
                            const caf::actor& object_storage_actor,
                            int node_rank,
                            BroadcastInitType init_type,
                            int batch_id,
                            const uint32_t tag) {
  assert(init_type == BroadcastInitType::kScatter || init_type == BroadcastInitType::kScatterFeatureOnly);

  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  auto recv_fn = [&](const std::string name, int tag, bool is_brecv) {
    auto actor = self->spawn(_recv_and_move_one_tensor,
                             mpi_actor,
                             object_storage_actor,
                             batch_id,
                             name,
                             tag,
                             is_brecv);
    return self->request(actor, caf::infinite, caf::get_atom_v);
  };

  if (init_type == BroadcastInitType::kScatter) {
    rh_list.push_back(recv_fn("new_gnids", tag, true));
    rh_list.push_back(recv_fn("src_gnids", tag + 1, true));
    rh_list.push_back(recv_fn("dst_gnids", tag + 2, true));
  }

  rh_list.push_back(recv_fn("new_features", tag + 3 + node_rank, false));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  self->receive([](caf::get_atom) {});
}

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   const caf::actor& trace_actor,
                   int node_rank,
                   int batch_id,
                   const NDArray& new_gnids,
                   const NDArray& new_features,
                   const NDArray& src_gnids,
                   const NDArray& dst_gnids,
                   const uint32_t tag) {
  // TraceMe send(batch_id, "input_send");
  auto start_time = std::chrono::steady_clock::now();
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, new_gnids, tag));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, src_gnids, tag + 1));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, dst_gnids, tag + 2));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, new_features, tag + 3));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  if (TRACE_ENABLED) {
    int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
    self->send(trace_actor, caf::put_atom_v, batch_id, "input_send", elapsed);
  }
}

void input_recv_fn(caf::blocking_actor* self,
                   const caf::actor& mpi_actor,
                   const caf::actor& object_storage_actor,
                   int batch_id,
                   const uint32_t tag) {
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  auto recv_fn = [&](const std::string name, int tag) {
    auto actor = self->spawn(_recv_and_move_one_tensor,
                             mpi_actor,
                             object_storage_actor,
                             batch_id,
                             name,
                             tag,
                             false);
    return self->request(actor, caf::infinite, caf::get_atom_v);
  };

  rh_list.push_back(recv_fn("new_gnids", tag));
  rh_list.push_back(recv_fn("src_gnids", tag + 1));
  rh_list.push_back(recv_fn("dst_gnids", tag + 2));
  rh_list.push_back(recv_fn("new_features", tag + 3));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  self->receive([](caf::get_atom) {});
}

void move_input_to_shared_mem_fn(caf::blocking_actor *self,
                                 const caf::actor& object_storage_actor,
                                 const caf::actor& trace_actor,
                                 int batch_id,
                                 const NDArray& new_gnids,
                                 const NDArray& new_features,
                                 const NDArray& src_gnids,
                                 const NDArray& dst_gnids) {
  // TraceMe move_input(batch_id, "input_send");
  auto start_time = std::chrono::steady_clock::now();
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  auto fn = [&](const std::string& name, const NDArray& arr) {
    auto actor = self->spawn(_move_input_to_shared_mem_one_tensor, object_storage_actor, name, arr);
    auto rh = self->request(actor, caf::infinite, caf::get_atom_v);
    return rh;
  };

  rh_list.push_back(fn("new_gnids", new_gnids));
  rh_list.push_back(fn("new_features", new_features));
  rh_list.push_back(fn("src_gnids", src_gnids));
  rh_list.push_back(fn("dst_gnids", dst_gnids));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  if (TRACE_ENABLED) {
    int elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
    self->send(trace_actor, caf::put_atom_v, batch_id, "input_send", elapsed);
  }
  self->receive([](caf::get_atom) { }); 
}

void sampling_fn(caf::blocking_actor* self,
                 const caf::actor& sampler,
                 int batch_id) {
  auto rh = self->request(sampler, caf::infinite, caf::sampling_atom_v, batch_id);
  receive_result<bool>(rh);
  self->receive([](caf::get_atom) { });
}

void cleanup_fn(caf::blocking_actor* self,
                const caf::actor& sampler,
                int batch_id) {
  auto rh = self->request(sampler, caf::infinite, caf::cleanup_atom_v, batch_id);
  receive_result<bool>(rh);
  self->receive([](caf::get_atom) { });
}

void gnn_execute_fn(caf::blocking_actor *self,
                    const caf::actor& gnn_executor_grp_actor,
                    int batch_id,
                    gnn_executor_request_type req_type,
                    int local_rank,
                    int param0) {
  auto rh = self->request(gnn_executor_grp_actor,
                          caf::infinite,
                          caf::exec_atom_v,
                          batch_id,
                          static_cast<int>(req_type),
                          local_rank,
                          param0);
  receive_result<bool>(rh);
  self->receive([](caf::get_atom) { });
}

void gnn_broadcast_execute_fn(caf::blocking_actor *self,
                              const caf::actor& gnn_executor_grp_actor,
                              int batch_id,
                              gnn_executor_request_type req_type,
                              int param0) {
  auto rh = self->request(gnn_executor_grp_actor,
                          caf::infinite,
                          caf::broadcast_exec_atom_v,
                          batch_id,
                          static_cast<int>(req_type),
                          param0);
  receive_result<bool>(rh);
  self->receive([](caf::get_atom) { });
}

}
}
