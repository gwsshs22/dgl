/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/trace_gen_helper.h
 * \brief 
 */
#pragma once

#include <tuple>
#include <vector>

#include <dgl/base_heterograph.h>

namespace dgl {
namespace omega {

std::vector<IdArray> TraceGenHelper(
    int first_new_gnid,
    const IdArray& infer_target_mask,
    const IdArray& batch_local_ids,
    const IdArray& u,
    const IdArray& v,
    const IdArray& u_in_partitions,
    const IdArray& v_in_partitions);

}
}
