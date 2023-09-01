/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/sampler.h
 * \brief 
 */
#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>

namespace dgl {
namespace omega {

void PutTrace(int batch_id, const std::string& name, int elapsed_micro);

void WriteTraces(const std::string& breakdown_trace_dir, const std::string& file_name);

}
}
