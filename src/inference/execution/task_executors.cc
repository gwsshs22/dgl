#include "task_executors.h"

namespace dgl {
namespace inference {

void input_bsend_fn(caf::blocking_actor* self,
                    const caf::actor& mpi_actor,
                    const NDArray& new_ngids,
                    const NDArray& src_ngids,
                    const NDArray& dst_ngids) {
  auto fn = [&](const NDArray& arr) {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_bsend_atom_v, arr);
    receive_result<void>(rh);
  };
  
  fn(new_ngids);
  fn(src_ngids);
  fn(dst_ngids);

  self->receive([](caf::get_atom) {
  });
}

void input_brecv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor) {

  auto fn = [&]() {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_brecv_atom_v, 0);
    return receive_result<NDArray>(rh);
  };

  NDArray new_ngids = fn();
  NDArray src_ngids = fn();
  NDArray dst_ngids = fn();

  self->receive([=](caf::get_atom) {
    auto ret = std::vector<NDArray>();
    ret.push_back(std::move(new_ngids));
    ret.push_back(std::move(src_ngids));
    ret.push_back(std::move(dst_ngids));
    return ret;
  });
}

void input_send_fn(caf::blocking_actor *self,
                   const caf::actor& mpi_actor,
                   int node_rank,
                   const NDArray& new_ngids,
                   const NDArray& src_ngids,
                   const NDArray& dst_ngids) {
  auto fn = [&](const NDArray& arr) {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_send_atom_v, node_rank, arr);
    receive_result<void>(rh);
  };

  fn(new_ngids);
  fn(src_ngids);
  fn(dst_ngids);
}

void input_recv_fn(caf::blocking_actor* self, const caf::actor& mpi_actor) {
  auto fn = [&]() {
    auto rh = self->request(mpi_actor, caf::infinite, caf::mpi_recv_atom_v, 0);
    return receive_result<NDArray>(rh);
  };

  NDArray new_ngids = fn();
  NDArray src_ngids = fn();
  NDArray dst_ngids = fn();

  self->receive([=](caf::get_atom) {
    auto ret = std::vector<NDArray>();
    ret.push_back(std::move(new_ngids));
    ret.push_back(std::move(src_ngids));
    ret.push_back(std::move(dst_ngids));
    return ret;
  });
}

}
}
