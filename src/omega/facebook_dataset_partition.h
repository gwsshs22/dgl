/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/omega/facebook_dataset_partition.h
 * \brief 
 */
#pragma once

#include <tuple>
#include <vector>

#include <dgl/base_heterograph.h>

namespace dgl {
namespace omega {

void PartitionFacebookDataset(
  const int num_parts,
  const std::string& input_dir,
  const std::vector<std::string>& edge_file_paths,
  const bool include_out_edges,
  const double infer_prob,
  const int num_omp_threads
);

}
}
