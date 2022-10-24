#include "mpi_actor.h"

#include <gloo/broadcast.h>

#include "gloo_rendezvous_actor.h"

namespace dgl {
namespace inference {

mpi_actor::mpi_actor(caf::actor_config& config,
          const caf::strong_actor_ptr& control_actor_ptr,
          const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
          const MpiConfig& mpi_config)
    : event_based_actor(config),
      control_actor_ptr_(control_actor_ptr),
      gloo_rendezvous_actor_ptr_(gloo_rendezvous_actor_ptr),
      mpi_config_(mpi_config) {

  initializing_.assign(make_initializing());
  running_.assign(make_running());
}

caf::behavior mpi_actor::make_behavior() {
  return initializing_;
}

caf::behavior mpi_actor::make_initializing() {
  return {
    [=](caf::init_atom) {
      auto hostname = mpi_config_.hostname;
      auto iface = mpi_config_.iface;
      rank_ = mpi_config_.rank;
      world_size_ = mpi_config_.world_size;

      gloo::transport::tcp::attr attr;
      attr.hostname = hostname;
      attr.iface = iface;
      attr.ai_family = AF_UNSPEC;

      auto gloo_store = std::make_unique<ActorBasedStore>(this->system(), gloo_rendezvous_actor_ptr_);

      gloo_device_ = gloo::transport::tcp::CreateDevice(attr);  
      gloo_context_ = std::make_shared<gloo::rendezvous::Context>(rank_, world_size_);
      gloo_context_->connectFullMesh(*gloo_store, gloo_device_);

      auto control_actor = caf::actor_cast<caf::actor>(control_actor_ptr_);
      send(control_actor, caf::initialized_atom_v,
          MpiInitMsg { caf::actor_cast<caf::strong_actor_ptr>(this), rank_, world_size_ });

      become(running_);
    }
  };
}

caf::behavior mpi_actor::make_running() {
  return {
    [=](int) {
  
      int data_arr[] = { 0, 0, 0, 0 };
      if (rank_ == 0) {
        data_arr[0] = 1;
        data_arr[1] = 2;
        data_arr[2] = 3;
        data_arr[3] = 4;
      }

      auto broadcast_opts = gloo::BroadcastOptions(gloo_context_);
      broadcast_opts.setOutput(data_arr, 4);
      broadcast_opts.setRoot(0);
      broadcast_opts.setTag(0);
      gloo::broadcast(broadcast_opts);

      for (int i = 0; i < 4; i++) {
        std::cout << data_arr[i] << " ";
      }

      std::cout << std::endl;
    }
  };
}

}
}