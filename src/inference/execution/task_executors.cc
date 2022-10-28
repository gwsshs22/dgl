#include "task_executors.h"

namespace dgl {
namespace inference {

void input_broadcast_fn(caf::blocking_actor* self,
                        const caf::actor& mpi_actor,
                        const NDArray& new_ngids,
                        const NDArray& src_ngids,
                        const NDArray& dst_ngids) {
  auto fn = [&](const NDArray& arr) {
    auto rh = self->request(mpi_actor, caf::infinite, caf::broadcast_atom_v, arr);
    receive_result<void>(rh);
  };
  
  fn(new_ngids);
  fn(src_ngids);
  fn(dst_ngids);

  self->receive([](caf::get_atom) {
  });
}

void input_receive_fn(caf::blocking_actor* self, const caf::actor& mpi_actor) {

  auto fn = [&]() {
    auto rh = self->request(mpi_actor, caf::infinite, caf::receive_atom_v, 0);
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
