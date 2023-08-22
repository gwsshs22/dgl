/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/sampling.h
 * \brief 
 */
#pragma once

#include <utility>

#include <dgl/base_heterograph.h>

namespace dgl {
namespace omega {

std::vector<std::pair<IdArray, IdArray>> SampleEdges(
    const IdArray& target_gnids,
    const IdArray& src_gnids,
    const IdArray& dst_gnids,
    const std::vector<int64_t>& fanouts);

}
}
