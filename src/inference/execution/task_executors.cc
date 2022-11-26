#include "task_executors.h"
#include "mem_utils.h"

namespace dgl {
namespace inference {

void input_bsend_fn(caf::blocking_actor* self,
                    const caf::actor& mpi_actor,
                    int batch_id,
                    const NDArray& new_gnids,
                    const NDArray& new_features,
                    const NDArray& src_gnids,
                    const NDArray& dst_gnids,
                    const uint32_t tag) {
  TraceMe send(batch_id, "input_send");

  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, new_gnids, tag));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, new_features, tag + 1));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, src_gnids, tag + 2));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, dst_gnids, tag + 3));
  
  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }

  self->receive([](caf::get_atom) { });
}

void input_brecv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag) {
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, 0, tag));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, 0, tag + 1));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, 0, tag + 2));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, 0, tag + 3));


  auto ret = std::vector<NDArray>();
  for (int i = 0; i < rh_list.size(); i++) {
    ret.push_back(receive_result<NDArray>(rh_list[i]));
  }

  self->receive([=](caf::get_atom) {
    return ret;
  });
}

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   int node_rank,
                   int batch_id,
                   const NDArray& new_gnids,
                   const NDArray& new_features,
                   const NDArray& src_gnids,
                   const NDArray& dst_gnids,
                   const uint32_t tag) {
  TraceMe send(batch_id, "input_send");

  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, new_gnids, tag));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, new_features, tag + 1));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, src_gnids, tag + 2));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, dst_gnids, tag + 3));
  
  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<bool>(rh_list[i]);
  }
}

void input_recv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag) {
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, 0, tag));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, 0, tag + 1));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, 0, tag + 2));
  rh_list.push_back(self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, 0, tag + 3));

  auto ret = std::vector<NDArray>();
  for (int i = 0; i < rh_list.size(); i++) {
    ret.push_back(receive_result<NDArray>(rh_list[i]));
  }

  self->receive([=](caf::get_atom) {
    return ret;
  });
}

void move_input_to_shared_mem_one_tensor(caf::blocking_actor *self,
                                        const caf::actor& object_storage_actor,
                                        int batch_id,
                                        const std::string& name,
                                        const NDArray& arr) {
  auto rh = self->request(object_storage_actor, caf::infinite, caf::put_atom_v, name, arr);
  receive_result<bool>(rh);
  rh = self->request(object_storage_actor, caf::infinite, caf::move_to_shared_atom_v, name);
  receive_result<bool>(rh);

  self->receive([](caf::get_atom) {});
}

void move_input_to_shared_mem_fn(caf::blocking_actor *self,
                                 const caf::actor& object_storage_actor,
                                 int batch_id,
                                 const NDArray& new_gnids,
                                 const NDArray& new_features,
                                 const NDArray& src_gnids,
                                 const NDArray& dst_gnids) {
  TraceMe move_input(batch_id, "input_copy_to_shared_mem");
  std::vector<caf::response_handle<caf::blocking_actor, caf::message, true>> rh_list;

  auto fn = [&](const std::string& name, const NDArray& arr) {
    auto actor = self->spawn(move_input_to_shared_mem_one_tensor, object_storage_actor, batch_id, name, arr);
    auto rh = self->request(actor, caf::infinite, caf::get_atom_v);
    return rh;
  };

  rh_list.push_back(fn("new_gnids", new_gnids));
  rh_list.push_back(fn("new_features", new_features));
  rh_list.push_back(fn("src_gnids", src_gnids));
  rh_list.push_back(fn("dst_gnids", dst_gnids));

  for (int i = 0; i < rh_list.size(); i++) {
    receive_result<void>(rh_list[i]);
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
