#include "task_executors.h"
#include "mem_utils.h"

namespace dgl {
namespace inference {

void input_bsend_fn(caf::blocking_actor* self,
                    const caf::actor& mpi_actor,
                    const NDArray& new_gnids,
                    const NDArray& new_features,
                    const NDArray& src_gnids,
                    const NDArray& dst_gnids,
                    const uint32_t tag) {
  auto fn = [&](const NDArray& arr) {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, arr, tag);
    receive_result<bool>(rh);
  };

  fn(new_gnids);
  fn(new_features);
  fn(src_gnids);
  fn(dst_gnids);

  self->receive([](caf::get_atom) { });
}

void input_brecv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag) {

  auto fn = [&]() {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, 0, tag);
    return receive_result<NDArray>(rh);
  };

  NDArray new_gnids = fn();
  NDArray new_features = fn();
  NDArray src_gnids = fn();
  NDArray dst_gnids = fn();

  self->receive([=](caf::get_atom) {
    auto ret = std::vector<NDArray>();
    ret.push_back(std::move(new_gnids));
    ret.push_back(std::move(new_features));
    ret.push_back(std::move(src_gnids));
    ret.push_back(std::move(dst_gnids));
    return ret;
  });
}

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   int node_rank,
                   const NDArray& new_gnids,
                   const NDArray& new_features,
                   const NDArray& src_gnids,
                   const NDArray& dst_gnids,
                   const uint32_t tag) {
  auto fn = [&](const NDArray& arr) {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, arr, tag);
    receive_result<bool>(rh);
  };

  fn(new_gnids);
  fn(new_features);
  fn(src_gnids);
  fn(dst_gnids);
}

void input_recv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor, const uint32_t tag) {
  auto fn = [&]() {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, 0, tag);
    return receive_result<NDArray>(rh);
  };

  NDArray new_gnids = fn();
  NDArray new_features = fn();
  NDArray src_gnids = fn();
  NDArray dst_gnids = fn();

  self->receive([=](caf::get_atom) {
    auto ret = std::vector<NDArray>();
    ret.push_back(std::move(new_gnids));
    ret.push_back(std::move(new_features));
    ret.push_back(std::move(src_gnids));
    ret.push_back(std::move(dst_gnids));
    return ret;
  });
}

void move_input_to_shared_mem_fn(caf::blocking_actor *self,
                                 const caf::actor& object_storage_actor,
                                 const NDArray& new_gnids,
                                 const NDArray& new_features,
                                 const NDArray& src_gnids,
                                 const NDArray& dst_gnids) {
  auto fn = [&](const std::string& name, const NDArray& arr) {
    auto rh = self->request(object_storage_actor, caf::infinite, caf::put_atom_v, name, arr);
    receive_result<bool>(rh);
    rh = self->request(object_storage_actor, caf::infinite, caf::move_to_shared_atom_v, name);
    receive_result<bool>(rh);
  };

  fn("new_gnids", new_gnids);
  fn("new_features", new_features);
  fn("src_gnids", src_gnids);
  fn("dst_gnids", dst_gnids);

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
