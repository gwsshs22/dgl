#include <iostream>

#include <dgl/runtime/container.h>
#include "../c_api_common.h"

#include "distributed_block.h"

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("omega.omega_apis._CAPI_DGLOmegaToDistributedBlocks")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int num_machines = args[0];
    int machine_rank = args[1];
    int num_gpus_per_machine = args[2];
    IdArray target_gnids = args[3];
    IdArray src_gnids = args[4];
    IdArray src_part_ids = args[5];
    IdArray dst_gnids = args[6];

    auto results = omega::ToDistributedBlocks(
        num_machines,
        machine_rank,
        num_gpus_per_machine,
        target_gnids,
        src_gnids,
        src_part_ids,
        dst_gnids);

    CHECK_EQ(num_gpus_per_machine, results.first.size());
    CHECK_EQ(num_gpus_per_machine, results.second.size());

    List<runtime::Value> ret_list;
    for (int i = 0; i < num_gpus_per_machine; i++) {
      ret_list.push_back(runtime::Value(MakeValue(HeteroGraphRef(results.first[i]))));
      ret_list.push_back(runtime::Value(MakeValue(results.second[i])));
    }

    *rv = ret_list;
  });

}
