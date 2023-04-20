#include "object_storage_actor.h"

#include "mem_utils.h"

namespace dgl {
namespace inference {

// TODO: if object_storage_actor is destroyed while blocking actors are running, it will make segfault.
void move_to_shared_mem_fn(caf::blocking_actor* self,
                           const caf::actor& object_storage_actor,
                           int batch_id,
                           caf::response_promise rp,
                           std::string name,
                           NDArray src_arr,
                           bool metadata_shared_mem_exists) {

  bool src_in_shared_mem = src_arr.GetSharedMem() != nullptr;
  if (src_in_shared_mem) {
    if (metadata_shared_mem_exists) {
      rp.deliver(true);
      return;
    } else {
      auto rh = self->request(object_storage_actor, caf::infinite, caf::internal_set_atom_v, name, CreateMetadata(batch_id, name, src_arr));
      receive_result<bool>(rh);
      rp.deliver(true);
      return;
    }
  }

  NDArray copied = CopyToSharedMem(batch_id, name, src_arr);

  if (metadata_shared_mem_exists) {
    auto rh = self->request(object_storage_actor, caf::infinite, caf::internal_set_atom_v, name, copied);
    receive_result<bool>(rh);
    rp.deliver(true);
  } else {
    auto rh = self->request(object_storage_actor, caf::infinite, caf::internal_set_atom_v, name, copied, CreateMetadata(batch_id, name, src_arr));
    receive_result<bool>(rh);
    rp.deliver(true);
  }
}


// It turns out that it is impossible to share device memory between processes.
// This will not be called.
void move_to_gpu_mem_fn(caf::blocking_actor* self,
                        const caf::actor& object_storage_actor,
                        int batch_id,
                        caf::response_promise rp,
                        std::string name,
                        NDArray src_arr,
                        int local_rank,
                        bool metadata_shared_mem_exists) {

  if (src_arr->ctx.device_type == kDLGPU && src_arr->ctx.device_id == local_rank) {
    if (metadata_shared_mem_exists) {
      rp.deliver(true);
      return;
    } else {
      auto rh = self->request(object_storage_actor, caf::infinite, caf::internal_set_atom_v, name, CreateMetadata(batch_id, name, src_arr));
      receive_result<bool>(rh);
      rp.deliver(true);
      return;
    }
  }
  
  NDArray copied = CopyToGpu(src_arr, local_rank);

  if (metadata_shared_mem_exists) {
    auto rh = self->request(object_storage_actor, caf::infinite, caf::internal_set_atom_v, name, copied);
    receive_result<bool>(rh);
    rp.deliver(true);
  } else {
    auto rh = self->request(object_storage_actor, caf::infinite, caf::internal_set_atom_v, name, copied, CreateMetadata(batch_id, name, src_arr));
    receive_result<bool>(rh);
    rp.deliver(true);
  }
}

caf::behavior object_storage_actor(caf::stateful_actor<object_storage>* self,
                                   int batch_id) {
  self->state.batch_id = batch_id;
  return {
    [=](caf::put_atom, const std::string& name, const NDArray& arr) {
      assert(self->state.arrays.find(name) == self->state.arrays.end());
      self->state.arrays[name] = arr;
      return true;
    },
    [=](caf::put_atom, const std::string& name, const NDArrayWithSharedMeta& p) {
      assert(self->state.arrays.find(name) == self->state.arrays.end());
      self->state.arrays[name] = p.first;
      self->state.metadata_shared_mems[name] = p.second;
      return true;
    },
    [=](caf::get_atom, const std::string& name) {
      assert(self->state.arrays.find(name) != self->state.arrays.end());
      return self->state.arrays[name];
    },
    [=](caf::delete_atom, const std::string& name) {
      auto array_it = self->state.arrays.find(name);
      if (array_it != self->state.arrays.end()) {
        self->state.arrays.erase(array_it);
      }

      auto meta_it = self->state.metadata_shared_mems.find(name);
      if (meta_it != self->state.metadata_shared_mems.end()) {
        self->state.metadata_shared_mems.erase(meta_it);
      }
      return true;
    },
    [=](caf::move_to_shared_atom, const std::string& name) {
      assert(self->state.arrays.find(name) != self->state.arrays.end());
      auto src_arr = self->state.arrays[name];
      bool metadata_shared_mem_exists =
          self->state.metadata_shared_mems.find(name) != self->state.metadata_shared_mems.end();

      auto rp = self->make_response_promise<bool>();
      self->spawn(move_to_shared_mem_fn, self, batch_id, rp, name, src_arr, metadata_shared_mem_exists);
      return rp;
    },
    [=](caf::move_to_gpu_atom, const std::string& name, int local_rank) {
      // It turns out that it is impossible to share device memory between processes.
      CHECK(false);

      assert(self->state.arrays.find(name) != self->state.arrays.end());
      auto src_arr = self->state.arrays[name];
      bool metadata_shared_mem_exists =
          self->state.metadata_shared_mems.find(name) != self->state.metadata_shared_mems.end();

      auto rp = self->make_response_promise<bool>();
      self->spawn(move_to_gpu_mem_fn, self, batch_id, rp, name, src_arr, local_rank, metadata_shared_mem_exists);
      return rp;
    },
    [=](caf::internal_set_atom,
        const std::string& name,
        const NDArray& arr,
        const std::shared_ptr<runtime::SharedMemory>& shared_mem) {
      self->state.arrays[name] = arr;
      self->state.metadata_shared_mems[name] = shared_mem;
      return true;
    },
    [=](caf::internal_set_atom,
        const std::string& name,
        const NDArray& arr) {
      self->state.arrays[name] = arr;
      return true;
    },
    [=](caf::internal_set_atom,
        const std::string& name,
        const std::shared_ptr<runtime::SharedMemory>& shared_mem) {
      self->state.metadata_shared_mems[name] = shared_mem;
      return true;
    }
  };
};

}
}
