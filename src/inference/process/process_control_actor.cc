#include "process_control_actor.h"

#include <sys/prctl.h>
#include <linux/limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <string>

#include <dgl/inference/envs.h>

namespace dgl {
namespace inference {

caf::behavior process_monitor_fn(caf::stateful_actor<process_monitor_state>* self) {
  return {
    [=](caf::get_atom) {
      int new_actor_process_global_id = (self->state.global_actor_process_id_counter)++;
      self->state.proc_ctrl_actor_map.insert(std::make_pair(new_actor_process_global_id, self->current_sender()));
      return new_actor_process_global_id;
    },
    [=](caf::initialized_atom,
        int actor_process_global_id) {
      auto it = self->state.proc_ctrl_actor_map.find(actor_process_global_id);
      if (it == self->state.proc_ctrl_actor_map.end()) {
        // TODO: error handling
        return;
      }

      auto process_control_actor = caf::actor_cast<caf::actor>((*it).second);
      self->state.proc_ctrl_actor_map.erase(it);

      // current_sender() should be an actor created with 'process_request_handler_fn' in a newly created actor process.
      self->send(process_control_actor, caf::initialized_atom_v, self->current_sender());
    }
  };
}

void process_creator_fn(caf::blocking_actor* self) {
  bool running = true;

  self->receive_while([&] { return running; }) (
    [&](caf::create_atom, int actor_process_global_id, const EnvSetter& setter) {
      auto sender = caf::actor_cast<caf::actor>(self->current_sender());
      ForkActorProcess(actor_process_global_id, setter);
      self->send(sender, caf::done_atom_v);
    },
    [&](caf::exit_msg& exit) {
      running = false;
    },
    [&](caf::error& err) {
      running = false;
    }
  );
}

void ForkActorProcess(int actor_process_global_id, const EnvSetter& env_setter) {
  pid_t parent_pid = getpid();
  pid_t pid = fork();
  if (pid == -1) {
    std::cerr << "Failed to fork a process: " <<
        strerror(errno) << std::endl;
    // TODO: error handling
    exit(-1);
  }

  if (pid == 0) {
    char exec_file_path[PATH_MAX + 1];
    int ret;

    // The child process will be died when parent's thread is exited.
    ret = prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (ret == -1) {
      std::cerr << "Set prctl(PR_SET_PDEATHSIG, SIGTERM) failed: " <<
          strerror(errno) << std::endl;
      exit(-1);
    }

    // The parent thread is exited and re-parenting occurred.
    if (getppid() != parent_pid) {
      std::cerr << "Parent is already dead: " <<
          strerror(errno) << std::endl;
      exit(-1);
    }

    ret = readlink("/proc/self/exe", exec_file_path, PATH_MAX);
    if(ret == -1) {
      std::cerr << "Cannot read current process exec file name: " <<
          strerror(errno) << std::endl;
      // TODO: error handling
      exit(-1);
    }
    exec_file_path[ret] = 0;

    SetEnv(DGL_INFER_ACTOR_PROCESS_GLOBAL_ID, actor_process_global_id);
    env_setter();
    
    // ${PYTHONBIN} -m dgl.inference.main
    ret = execlp(exec_file_path,
      exec_file_path,
      "-m",
      "dgl.inference.fork",
      (char *)NULL);

    if (ret < 0) {
      std::cerr << "Failed to exec a new child process with the executable ( " <<
          exec_file_path << ") : " << strerror(errno) << std::endl;
      // TODO: error handling
      exit(-1);
    }
    // never reached
  }
}

caf::behavior process_request_handler_fn(caf::event_based_actor* self, int actor_process_global_id) {
  auto process_monitor_ptr = self->system().registry().get(caf::process_mon_atom_v);
  auto process_monitor = caf::actor_cast<caf::actor>(process_monitor_ptr);
  self->send(process_monitor, caf::initialized_atom_v, actor_process_global_id);

  return {
    [](int x) {
      std::cerr << "A process request handler got " << x << std::endl;
    }
  };
}

// process_control_actor

process_control_actor::process_control_actor(caf::actor_config& config,
                                             const caf::strong_actor_ptr& owner_ptr,
                                             const std::string& role,
                                             int local_rank)
    : event_based_actor(config), owner_ptr_(owner_ptr), role_(role), local_rank_(local_rank) {
}

caf::behavior process_control_actor::make_behavior() {
  send(this, caf::init_atom_v);
  return {
    [&](caf::init_atom) {
      auto process_monitor_ptr = system().registry().get(caf::process_mon_atom_v);
      auto process_monitor = caf::actor_cast<caf::actor>(process_monitor_ptr);
      auto process_creator_ptr = system().registry().get(caf::process_creator_atom_v);
      request(process_monitor, caf::infinite, caf::get_atom_v).then(
        [=](int global_actor_process_id) {
          auto process_creator = caf::actor_cast<caf::actor>(process_creator_ptr);
          send(process_creator, caf::create_atom_v, global_actor_process_id, MakeEnvSetter());
        },
        [&](caf::error& err) {
          // TODO: error handling
          caf::aout(this) << "Error: " << caf::to_string(err) << std::endl;
        }
      );
    },
    // From process_creator
    [&](caf::done_atom) {
    },
    // From process_monitor
    [&](caf::initialized_atom, const caf::strong_actor_ptr& process_request_handler) {
      auto owner = caf::actor_cast<caf::actor>(owner_ptr_);
      send(owner, caf::initialized_atom_v, role_, local_rank_);

      // process_request_handler is in the newly created actor process
      become(make_running_behavior(process_request_handler));

    }
  };
}

}
}
