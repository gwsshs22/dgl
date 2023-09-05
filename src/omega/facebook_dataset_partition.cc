#include "facebook_dataset_partition.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <utility>
#include <chrono>

#include <dgl/runtime/ndarray.h>
#include <dgl/aten/array_ops.h>
#include <dgl/aten/types.h>
#include <dgl/runtime/parallel_for.h>
#include <dmlc/omp.h>
#include <parallel_hashmap/phmap.h>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace omega {

constexpr uint64_t kDGLSerialize_Tensors = 0xDD5A9FBE3FA2443F;
typedef std::pair<std::string, NDArray> NamedTensor;
typedef phmap::flat_hash_map<int64_t, int64_t> IdMap;
typedef phmap::flat_hash_set<int64_t, int64_t> IdSet;

inline uint64_t hash(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;
}

inline bool isInferTarget(int x, int infer_target_interval) {
  return hash(x) % infer_target_interval == 0;
}

int BuildIdMappings(
  const std::vector<std::string>& edge_file_paths,
  std::vector<IdMap>& orig_to_news,
  std::vector<std::vector<int64_t>>& new_to_origs,
  std::vector<int64_t>& num_edges_per_file) {
  
  const int num_parts = orig_to_news.size();
  int max_node_id = -1;
  int edge_file_idx = 0;
  for (const auto& edge_file_path : edge_file_paths) {
    std::ifstream input_stream(edge_file_path);
    std::string line;
    while (std::getline(input_stream, line)) {
      std::istringstream is(line);

      int dst_id;
      is >> dst_id;
      max_node_id = std::max(dst_id, max_node_id);
      int part_id = hash(dst_id) % num_parts;

      auto ret = orig_to_news[part_id].insert({ dst_id, orig_to_news[part_id].size() });
      CHECK(ret.second);
      new_to_origs[part_id].push_back(dst_id);

      int64_t num_edges = 1 + 2 * (std::count(line.begin(), line.end(), ',') + 1);
      num_edges_per_file[edge_file_idx] += num_edges;
    }

    edge_file_idx++;
  }

  int num_nodes_sum = 0;
  std::cout << "Max node id = " << max_node_id << std::endl;
  for (int i = 0; i < num_parts; i++) {
    CHECK_EQ(orig_to_news[i].size(), new_to_origs[i].size());
    num_nodes_sum += orig_to_news[i].size();
    std::cout << "Partition-" << i << " " << orig_to_news[i].size() << " nodes assigned." << std::endl;
  }

  return max_node_id;
}

void SaveNdTensors(std::string filename, std::vector<NamedTensor> named_tensors) {
  auto fs = std::unique_ptr<dmlc::Stream>(
    dmlc::Stream::Create(filename.c_str(), "w"));
  CHECK(fs) << "Filename is invalid";
  fs->Write(kDGLSerialize_Tensors);
  
  fs->Write(static_cast<uint64_t>(named_tensors.size()));
  fs->Write(named_tensors);
}

std::pair<IdArray, IdArray> LoadNdEdges(std::string filename) {
  auto fs = std::unique_ptr<dmlc::Stream>(
      dmlc::Stream::Create(filename.c_str(), "r"));
  CHECK(fs) << "Filename is invalid or file doesn't exists";
  uint64_t magic_number;
  CHECK(fs->Read(&magic_number)) << "Invalid file";
  CHECK_EQ(magic_number, kDGLSerialize_Tensors);
  
  uint64_t num_tensors;
  fs->Read(&num_tensors);

  std::vector<NamedTensor> named_tensors;
  fs->Read(&named_tensors);

  CHECK_EQ(num_tensors, 2);
  CHECK_EQ(named_tensors.size(), 2);

  CHECK_EQ(named_tensors[0].first, "src_ids");
  CHECK_EQ(named_tensors[1].first, "dst_ids");

  return std::make_pair(named_tensors[0].second, named_tensors[1].second);
}

void SplitEdges(
  const int num_parts,
  const std::string& input_dir,
  const std::string& edge_file_path,
  const int edge_file_idx,
  const int64_t num_edges,
  const int infer_target_interval) {

  auto srcs = aten::NewIdArray(num_edges);
  auto dsts = aten::NewIdArray(num_edges);

  auto srcs_ptr = srcs.Ptr<int64_t>();
  auto dsts_ptr = dsts.Ptr<int64_t>();
  int itr = 0;

  std::ifstream input_stream(edge_file_path);
  std::string line;
  
  while (std::getline(input_stream, line)) {
    std::istringstream is(line);
    int dst_id;
    is >> dst_id;

    srcs_ptr[itr] = dst_id;
    dsts_ptr[itr++] = dst_id;

    while (!is.eof()) {
      int src_id;
      char tmp;
      is >> src_id;

      srcs_ptr[itr] = src_id;
      dsts_ptr[itr++] = dst_id;

      srcs_ptr[itr] = dst_id;
      dsts_ptr[itr++] = src_id;

      if (is.peek() != ',') {
        break;
      } else {
        is >> tmp;
      }
    }
  }

  CHECK_EQ(num_edges, itr);

  const int num_threads = runtime::compute_num_threads(0, num_edges, 1);
  std::vector<std::vector<int64_t>> p_sum(num_threads + 1);

  for (int t = 0; t < num_threads + 1; t++) {
    for (int p = 0; p < num_parts + 1; p++) {
      p_sum[t].push_back(0);
    }
  }

  std::vector<IdArray> srcs_in_parts(num_parts + 1);
  std::vector<int64_t*> src_ptrs_in_parts(num_parts + 1);
  std::vector<IdArray> dsts_in_parts(num_parts + 1);
  std::vector<int64_t*> dst_ptrs_in_parts(num_parts + 1);

#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();

    const int64_t start_i =
        thread_id * (num_edges / num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_edges % num_threads);
    const int64_t end_i =
        (thread_id + 1) * (num_edges / num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_edges % num_threads);
    assert(thread_id + 1 < num_threads || end_i == num_edges);

    for (int64_t i = start_i; i < end_i; i++) {
      auto src_id = srcs_ptr[i];
      auto dst_id = dsts_ptr[i];

      if (isInferTarget(src_id, infer_target_interval) || isInferTarget(dst_id, infer_target_interval)) {
        p_sum[thread_id + 1][num_parts]++;
      } else {
        int part_id = hash(dst_id) % num_parts;
        p_sum[thread_id + 1][part_id]++;
      }
    }
#pragma omp barrier
#pragma omp master
    {
      int64_t total_sum = 0;
      for (int p = 0; p < num_parts + 1; p++) {
        int64_t part_sum = 0;
        for (int t = 0; t < num_threads; t++) {
          p_sum[t + 1][p] += part_sum;
          part_sum = p_sum[t + 1][p];
        }

        total_sum += part_sum;
      }

      CHECK_EQ(total_sum, num_edges);

      for (int p = 0; p < num_parts + 1; p++) {
        srcs_in_parts[p] = aten::NewIdArray(p_sum[num_threads][p]);
        dsts_in_parts[p] = aten::NewIdArray(p_sum[num_threads][p]);

        src_ptrs_in_parts[p] = srcs_in_parts[p].Ptr<int64_t>();
        dst_ptrs_in_parts[p] = dsts_in_parts[p].Ptr<int64_t>();
      }
    }

#pragma omp barrier
    std::vector<int64_t> idx_arr(p_sum[thread_id]);

    for (int64_t i = start_i; i < end_i; i++) {
      auto src_id = srcs_ptr[i];
      auto dst_id = dsts_ptr[i];

      if (isInferTarget(src_id, infer_target_interval) || isInferTarget(dst_id, infer_target_interval)) {
        int64_t idx = idx_arr[num_parts]++;

        src_ptrs_in_parts[num_parts][idx] = src_id;
        dst_ptrs_in_parts[num_parts][idx] = dst_id;
      } else {
        int part_id = hash(dst_id) % num_parts;
        int64_t idx = idx_arr[part_id]++;

        src_ptrs_in_parts[part_id][idx] = src_id;
        dst_ptrs_in_parts[part_id][idx] = dst_id;
      }
    }

    for (int p = 0; p < num_parts + 1; p++) {
      CHECK_EQ(idx_arr[p], p_sum[thread_id + 1][p]);
    }
  }

  for (int p = 0; p < num_parts + 1; p++) {
    auto named_nd_tensors = std::vector<NamedTensor>();
    named_nd_tensors.emplace_back("src_ids", srcs_in_parts[p]);
    named_nd_tensors.emplace_back("dst_ids", dsts_in_parts[p]);
    SaveNdTensors(input_dir + "/edges-" + std::to_string(edge_file_idx) + "-part-" + std::to_string(p) + ".dgl", named_nd_tensors);
  }
}

void BuildPartData(
  const int max_node_id,
  const int num_total_nodes,
  const int dst_part_id,
  const int num_edge_files,
  const std::string& input_dir,
  std::vector<std::vector<int64_t>>& new_to_origs
) {

  const int num_parts = new_to_origs.size();

  std::vector<int64_t> part_id_offsets(num_parts, 0);
  for (int i = 1; i < num_parts; i++) {
    part_id_offsets[i] = part_id_offsets[i - 1] + new_to_origs[i - 1].size();
  }

  IdArray orig_to_local_mapping = aten::NewIdArray(max_node_id + 1);
  IdArray dgl_nids = aten::NewIdArray(num_total_nodes);
  IdArray part_ids = aten::NewIdArray(num_total_nodes);
  int64_t* dgl_nids_ptr = dgl_nids.Ptr<int64_t>();
  int64_t* part_ids_ptr = part_ids.Ptr<int64_t>();
  int64_t* orig_to_local_mapping_ptr = orig_to_local_mapping.Ptr<int64_t>();

  int local_id = 0;
  for (int i = 0; i < new_to_origs[dst_part_id].size(); i++) {
    auto orig_id = new_to_origs[dst_part_id][i];
    orig_to_local_mapping_ptr[orig_id] = local_id;
    dgl_nids_ptr[local_id] = part_id_offsets[dst_part_id] + i;
    part_ids_ptr[local_id] = dst_part_id;
    local_id++;
  }

  for (int p = 0; p < num_parts; p++) {
    if (p == dst_part_id) continue;

    for (int i = 0; i < new_to_origs[p].size(); i++) {
      auto orig_id = new_to_origs[p][i];
      orig_to_local_mapping_ptr[orig_id] = local_id;
      dgl_nids_ptr[local_id] = part_id_offsets[p] + i;
      part_ids_ptr[local_id] = p;
      local_id++;
    }
  }

  std::vector<IdArray> src_ids_vector;
  std::vector<IdArray> dst_ids_vector;

  for (int f = 0; f < num_edge_files; f++) {
    auto file_path = input_dir + "/edges-" + std::to_string(f) + "-part-" + std::to_string(dst_part_id) + ".dgl";
    auto load_ret = LoadNdEdges(file_path);
    src_ids_vector.push_back(load_ret.first);
    dst_ids_vector.push_back(load_ret.second);
  }

  IdArray src_ids = aten::Concat(src_ids_vector);
  IdArray dst_ids = aten::Concat(dst_ids_vector);

  const int64_t* src_ids_ptr = src_ids.Ptr<int64_t>();
  const int64_t* dst_ids_ptr = dst_ids.Ptr<int64_t>();

  const int64_t num_edges = src_ids->shape[0];

  IdArray new_src_ids = aten::NewIdArray(num_edges);
  IdArray new_dst_ids = aten::NewIdArray(num_edges);

  int64_t* new_src_ids_ptr = new_src_ids.Ptr<int64_t>();
  int64_t* new_dst_ids_ptr = new_dst_ids.Ptr<int64_t>();

  runtime::parallel_for(0, num_edges, [&](auto b, auto e) {
    for (auto i = b; i < e; i++) {
      const int64_t src_id = src_ids_ptr[i];
      const int64_t dst_id = dst_ids_ptr[i];

      CHECK_EQ(hash(dst_id) % num_parts, dst_part_id);

      new_src_ids_ptr[i] = orig_to_local_mapping_ptr[src_id];
      new_dst_ids_ptr[i] = orig_to_local_mapping_ptr[dst_id];
    }
  });

  auto named_nd_tensors = std::vector<NamedTensor>();
  named_nd_tensors.emplace_back("new_src_ids", new_src_ids);
  named_nd_tensors.emplace_back("new_dst_ids", new_dst_ids);
  named_nd_tensors.emplace_back("part_id", part_ids);
  named_nd_tensors.emplace_back("dgl_nids", dgl_nids);

  SaveNdTensors(input_dir + "/graph-data-" + std::to_string(dst_part_id) + ".dgl", named_nd_tensors);
}

void BuildInferTargetGraph(
  const int num_parts,
  const int max_node_id,
  const int num_total_nodes,
  const int infer_target_interval,
  const int num_edge_files,
  const std::string& input_dir,
  const IdArray& pid_to_orig_ids) {
  
  std::vector<IdArray> src_ids_vector;
  std::vector<IdArray> dst_ids_vector;

  for (int f = 0; f < num_edge_files; f++) {
    auto file_path = input_dir + "/edges-" + std::to_string(f) + "-part-" + std::to_string(num_parts) + ".dgl";
    auto load_ret = LoadNdEdges(file_path);
    src_ids_vector.push_back(load_ret.first);
    dst_ids_vector.push_back(load_ret.second);
  }

  IdArray src_ids = aten::Concat(src_ids_vector);
  IdArray dst_ids = aten::Concat(dst_ids_vector);

  const int64_t* src_ids_ptr = src_ids.Ptr<int64_t>();
  const int64_t* dst_ids_ptr = dst_ids.Ptr<int64_t>();
  const int64_t* pid_to_orig_ids_ptr = pid_to_orig_ids.Ptr<int64_t>();

  const int64_t num_edges = src_ids->shape[0];

  IdArray new_src_ids = aten::NewIdArray(num_edges);
  IdArray new_dst_ids = aten::NewIdArray(num_edges);
  IdArray orig_id_to_pids = aten::NewIdArray(max_node_id + 1);

  int64_t* new_src_ids_ptr = new_src_ids.Ptr<int64_t>();
  int64_t* new_dst_ids_ptr = new_dst_ids.Ptr<int64_t>();
  int64_t* orig_id_to_pids_ptr = orig_id_to_pids.Ptr<int64_t>();



  runtime::parallel_for(0, num_total_nodes, [&](auto b, auto e) {
    for (auto i = b; i < e; i++) {
      auto orig_id = pid_to_orig_ids_ptr[i];
      orig_id_to_pids_ptr[orig_id] = i;
    }
  });

  runtime::parallel_for(0, num_edges, [&](auto b, auto e) {
    for (auto i = b; i < e; i++) {
      auto src_id = src_ids_ptr[i];
      auto dst_id = dst_ids_ptr[i];

      new_src_ids_ptr[i] = orig_id_to_pids_ptr[src_id];
      new_dst_ids_ptr[i] = orig_id_to_pids_ptr[dst_id];
    }
  });

  IdArray infer_target_mask = aten::NewIdArray(num_total_nodes);
  int64_t* infer_target_mask_ptr = infer_target_mask.Ptr<int64_t>();

  runtime::parallel_for(0, num_total_nodes, [&](auto b, auto e) {
    for (auto i = b; i < e; i++) {
      auto orig_id = pid_to_orig_ids_ptr[i];
      if (isInferTarget(orig_id, infer_target_interval)) {
        infer_target_mask_ptr[i] = 1;
      } else {
        infer_target_mask_ptr[i] = 0;
      }
    }
  });

  auto named_nd_tensors = std::vector<NamedTensor>();
  named_nd_tensors.emplace_back("new_src_ids", new_src_ids);
  named_nd_tensors.emplace_back("new_dst_ids", new_dst_ids);
  named_nd_tensors.emplace_back("infer_target_mask", infer_target_mask);
  named_nd_tensors.emplace_back("dgl_nids", pid_to_orig_ids);

  SaveNdTensors(input_dir + "/infer-target-graph-data.dgl", named_nd_tensors);
}

void PartitionFacebookDataset(
  const int num_parts,
  const std::string& input_dir,
  const std::vector<std::string>& edge_file_paths,
  const double infer_prob,
  const int num_omp_threads
) {
  CHECK_GE(infer_prob, 0.0);
  CHECK_GE(1.0, infer_prob);

  const int infer_target_interval = (int)(1 / infer_prob);
  CHECK_GE(infer_target_interval, 1);

  omp_set_num_threads(num_omp_threads);
  std::vector<IdMap> orig_to_news(num_parts);
  std::vector<std::vector<int64_t>> new_to_origs(num_parts);
  std::vector<int64_t> num_edges_per_file(edge_file_paths.size(), 0);

  const int max_node_id = BuildIdMappings(edge_file_paths, orig_to_news, new_to_origs, num_edges_per_file);
  int num_total_nodes = 0;
  for (int p = 0; p < num_parts; p++) {
    num_total_nodes += new_to_origs[p].size();
  }

  int i = 0;
  for (const auto& edge_file_path : edge_file_paths) {
    SplitEdges(num_parts, input_dir, edge_file_path, i, num_edges_per_file[i], infer_target_interval);
    i++;
  }

  for (int p = 0; p < num_parts; p++) {
    BuildPartData(max_node_id, num_total_nodes, p, edge_file_paths.size(), input_dir, new_to_origs);
  }


  IdArray pid_to_orig_ids;
  IdArray infer_target_mask;
  {
    std::vector<IdArray> new_to_orig_arrs;
    for (int p = 0; p < num_parts; p++) {
      new_to_orig_arrs.push_back(NDArray::FromVector(new_to_origs[p]));
    }

    pid_to_orig_ids = aten::Concat(new_to_orig_arrs);
    infer_target_mask = aten::NewIdArray(num_total_nodes);
  }

  int64_t* infer_target_mask_ptr = infer_target_mask.Ptr<int64_t>();

  runtime::parallel_for(0, num_total_nodes, [&](auto b, auto e) {
    for (auto i = b; i < e; i++) {
      if (isInferTarget(i, infer_target_interval)) {
        infer_target_mask_ptr[i] = 1;
      } else {
        infer_target_mask_ptr[i] = 0;
      }
    }
  });

  BuildInferTargetGraph(
    num_parts,
    max_node_id,
    num_total_nodes,
    infer_target_interval,
    edge_file_paths.size(),
    input_dir,
    pid_to_orig_ids
  );

  std::vector<int64_t> num_nodes_per_parts;

  for (int p = 0; p < num_parts; p++) {
    num_nodes_per_parts.push_back(new_to_origs[p].size());
  }

  auto named_nd_tensors = std::vector<NamedTensor>();
  named_nd_tensors.emplace_back("infer_target_mask", infer_target_mask);
  named_nd_tensors.emplace_back("orig_ids", pid_to_orig_ids);
  named_nd_tensors.emplace_back("num_nodes_per_parts", NDArray::FromVector(num_nodes_per_parts));

  SaveNdTensors(input_dir + "/global-data.dgl", named_nd_tensors);
}

}
}
