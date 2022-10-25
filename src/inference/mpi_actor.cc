#include "mpi_actor.h"

#include <gloo/broadcast.h>

#include "gloo_rendezvous_actor.h"

namespace dgl {
namespace inference {

mpi_actor::mpi_actor(caf::actor_config& config,
          const caf::strong_actor_ptr& gloo_rendezvous_actor_ptr,
          const MpiConfig& mpi_config)
    : blocking_actor(config),
      gloo_rendezvous_actor_ptr_(gloo_rendezvous_actor_ptr),
      mpi_config_(mpi_config) {
}

void mpi_actor::act() {
  auto hostname = mpi_config_.hostname;
  auto iface = mpi_config_.iface;
  u_int32_t rank = mpi_config_.rank;
  u_int32_t world_size = mpi_config_.world_size;

  gloo::transport::tcp::attr attr;
  attr.hostname = hostname;
  attr.iface = iface;
  attr.ai_family = AF_UNSPEC;

  auto gloo_store = std::make_unique<ActorBasedStore>(this->system(), gloo_rendezvous_actor_ptr_);

  auto gloo_device = gloo::transport::tcp::CreateDevice(attr);  
  auto gloo_context = std::make_shared<gloo::rendezvous::Context>(rank, world_size);
  gloo_context->connectFullMesh(*gloo_store, gloo_device);

  bool running = true;
  auto init_mon_ptr = this->system().registry().get(caf::init_mon_atom_v);
  auto init_mon = caf::actor_cast<caf::actor>(init_mon_ptr);
  send(init_mon, caf::initialized_atom_v, "mpi", rank, world_size);

  receive_while([&]{ return running; }) (
    [=](int) {
  
      int data_arr[] = { 0, 0, 0, 0 };
      if (rank == 0) {
        data_arr[0] = 1;
        data_arr[1] = 2;
        data_arr[2] = 3;
        data_arr[3] = 4;
      }

      auto broadcast_opts = gloo::BroadcastOptions(gloo_context);
      broadcast_opts.setOutput(data_arr, 4);
      broadcast_opts.setRoot(0);
      broadcast_opts.setTag(0);
      gloo::broadcast(broadcast_opts);

      for (int i = 0; i < 4; i++) {
        std::cout << data_arr[i] << " ";
      }

      std::cout << std::endl;
    }
  );
}

}
}