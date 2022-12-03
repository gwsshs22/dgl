#include <dgl/inference/common.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/aten/types.h>
#include <dgl/runtime/parallel_for.h>
#include "../../src/array/array_op.h"
#include <algorithm>
#include <dmlc/omp.h>
#include <parallel_hashmap/phmap.h>

namespace dgl {
namespace inference {

constexpr uint64_t kDGLSerialize_Tensors = 0xDD5A9FBE3FA2443F;
typedef std::pair<std::string, NDArray> NamedTensor;

void CreateDirectory(std::string directory_path) {
  system(("mkdir -p " + directory_path).c_str());
}

void SaveNdTensors(std::string filename, std::vector<NamedTensor> named_tensors) {
  auto fs = std::unique_ptr<dmlc::Stream>(
    dmlc::Stream::Create(filename.c_str(), "w"));
  CHECK(fs) << "Filename is invalid";
  fs->Write(kDGLSerialize_Tensors);
  
  fs->Write(static_cast<uint64_t>(named_tensors.size()));
  fs->Write(named_tensors);
}

inline void AddEdge(std::vector<std::vector<int>>& src_ids_list,
                    std::vector<std::vector<int>>& dst_ids_list,
                    std::vector<int>& infer_src_ids,
                    std::vector<int>& infer_dst_ids,
                    int u,
                    int v,
                    int v_part_id,
                    bool is_infer_edge) {
  if (is_infer_edge) {
    infer_src_ids.push_back(u);
    infer_dst_ids.push_back(v);
  } else {
    src_ids_list[v_part_id].push_back(u);
    dst_ids_list[v_part_id].push_back(v);
  }
}

std::string GetIdxPostfix(int input_edge_partitions_idx) {
  std::string idx_postfix;
  if (input_edge_partitions_idx < 10) {
    idx_postfix = "0" + std::to_string(input_edge_partitions_idx);
  } else {
    idx_postfix = std::to_string(input_edge_partitions_idx);
  }

  return idx_postfix;
}

void SplitEdgesMain(int input_edge_partitions_idx,
                    int num_target_parts,
                    int inference_target_per,
                    std::vector<int>& max_node_ids,
                    std::vector<int>& num_src_nodes_list,
                    std::vector<int64_t>& num_edges_list,
                    std::string& base_dir,
                    std::string& output_dir) {

  std::string idx_postfix = GetIdxPostfix(input_edge_partitions_idx);
  std::string part_file_name = base_dir + "/part-m-000" + idx_postfix;

  auto src_ids_list = std::vector<std::vector<int>>();
  auto dst_ids_list = std::vector<std::vector<int>>();
  auto infer_src_ids = std::vector<int>();
  auto infer_dst_ids = std::vector<int>();

  for (int i = 0; i < num_target_parts; i++) {
    src_ids_list.push_back(std::vector<int>());
    dst_ids_list.push_back(std::vector<int>());
  }

  std::ifstream input_stream(part_file_name);

  std::string line;
  int max_node_id = -1;

  int num_src_nodes_in_part = 0;
  int num_edges_in_part = 0;

  while (std::getline(input_stream, line)) {
    std::istringstream is(line);
    std::string dest_nodes_str;
    int src_id;
    is >> src_id;
    is >> dest_nodes_str;
    num_src_nodes_in_part++;

    uint64_t src_hash = hash(src_id);
    int src_part_id = src_hash % num_target_parts;
    bool src_infer_target = hash(src_hash) % inference_target_per == 0;

    max_node_id = (max_node_id < src_id) ? src_id : max_node_id;

    size_t pos_start = 0, pos_end;
    std::string token;
    std::string delimiter = ",";

    AddEdge(src_ids_list, dst_ids_list, infer_src_ids, infer_dst_ids, src_id, src_id, src_part_id, src_infer_target);
    while ((pos_end = dest_nodes_str.find(delimiter, pos_start)) != std::string::npos) {
      num_edges_in_part += 2;
      token = dest_nodes_str.substr (pos_start, pos_end - pos_start);
      pos_start = pos_end + 1;
      int dst_id = std::stoi(token);
      max_node_id = (max_node_id < dst_id) ? dst_id : max_node_id;
      if (src_id == dst_id) {
        std::cerr << "Self loop is detected." << std::endl;
        exit(-1);
      }

      uint64_t dst_hash = hash(dst_id);
      int dst_part_id = dst_hash % num_target_parts;
      bool dst_infer_target = hash(dst_hash) % inference_target_per == 0;

      bool is_infer_edge = src_infer_target || dst_infer_target;

      AddEdge(src_ids_list, dst_ids_list, infer_src_ids, infer_dst_ids, src_id, dst_id, dst_part_id, is_infer_edge);
      AddEdge(src_ids_list, dst_ids_list, infer_src_ids, infer_dst_ids, dst_id, src_id, src_part_id, is_infer_edge);
    }
  }

  max_node_ids[input_edge_partitions_idx - 1] = max_node_id;
  num_src_nodes_list[input_edge_partitions_idx - 1] = num_src_nodes_in_part;
  num_edges_list[input_edge_partitions_idx - 1] = 2 * num_edges_in_part + num_src_nodes_in_part;

  for (int i = 0; i < num_target_parts; i++) {
    auto& src_ids = src_ids_list[i];
    auto& dst_ids = dst_ids_list[i];
    auto s = src_ids.size();
    auto output = std::ofstream(output_dir + "/edges_" + std::to_string(i) + "_" + idx_postfix + ".txt", std::ofstream::out | std::ofstream::trunc);
    for (int j = 0; j < s; j++) {
      output << src_ids[j] << ' ' << dst_ids[j] << std::endl;
    }
    output << -1 << std::endl;
  }

  auto infer_output = std::ofstream(output_dir + "/edges_infer_" + idx_postfix + ".txt", std::ofstream::out | std::ofstream::trunc);
  auto s = infer_src_ids.size();
  for (int j = 0; j < s; j++) {
    infer_output << infer_src_ids[j] << ' ' << infer_dst_ids[j] << std::endl;
  }
  infer_output << -1 << std::endl;
}

void SplitEdges(std::string base_dir, std::string output_dir, int num_input_edge_partitions, int num_target_parts, int infer_target_per) {
  auto max_node_ids = std::vector<int>(num_input_edge_partitions);
  auto num_src_nodes_list = std::vector<int>(num_input_edge_partitions);
  auto num_edges_list = std::vector<int64_t>(num_input_edge_partitions);

  auto threads = std::vector<std::thread>();

  for (int i = 1; i <= num_input_edge_partitions; i++) {
    threads.push_back(std::thread([&, i]() {
      SplitEdgesMain(i, num_target_parts, infer_target_per, max_node_ids, num_src_nodes_list, num_edges_list, base_dir, output_dir);
    }));
  }

  for (auto& t : threads) {
    t.join();
  }

  int max_node_id = -1;
  int num_src_nodes = 0;
  int64_t num_edges = 0;

  for (int i = 0; i < num_input_edge_partitions; i++) {
    max_node_id = (max_node_id < max_node_ids[i]) ? max_node_ids[i] : max_node_id;
    num_src_nodes += num_src_nodes_list[i];
    num_edges += num_edges_list[i];
  }

  std::cout << "max_node_id = " << max_node_id << ", num_src_nodes=" << num_src_nodes << ", num_edges=" << num_edges << std::endl;

  auto meta_output = std::ofstream(output_dir + "/max_node_id.txt", std::ofstream::out | std::ofstream::trunc);
  meta_output << max_node_id << std::endl;
}

void BuildInferTargetGraph(std::string output_dir, int num_input_edge_partitions, int num_target_parts, int infer_target_per) {
  std::cout << "BuildInferTargetGraph" << std::endl;
  auto u = std::vector<int64_t>();
  auto v = std::vector<int64_t>();

  auto new_to_old = std::vector<int64_t>();

  for (int i = 1; i <= num_input_edge_partitions; i++) {
    auto edge_file_path = output_dir + "/edges_infer_" + GetIdxPostfix(i) + ".txt";
    std::ifstream input_stream(edge_file_path);
    int src_id, dst_id;
    while (true) {
      input_stream >> src_id;
      if (src_id < 0) {
        break;
      }
      input_stream >> dst_id;
      u.push_back(src_id);
      v.push_back(dst_id);
    }
  }

  auto mapping = phmap::flat_hash_map<int64_t, int64_t>();
  for (auto& dst_id : v) {
    auto ret = mapping.insert({ dst_id, mapping.size() });
    if (ret.second) {
      new_to_old.push_back(dst_id);
    }
  }

  for (auto& src_id : u) {
    auto ret = mapping.insert({ src_id, mapping.size() });
    if (ret.second) {
      new_to_old.push_back(src_id);
    }
  }

  auto num_nodes = mapping.size();
  auto dgl_nid = IdArray::FromVector(new_to_old);

  auto infer_target_mask = aten::NewIdArray(num_nodes, DLContext{kDLCPU, 0}, 8);
  auto infer_target_mask_ptr = (int8_t*) infer_target_mask->data;

  runtime::parallel_for(0, num_nodes, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; i++) {
      auto& old_id = new_to_old[i];
      uint64_t old_id_hash = hash(old_id);
      bool is_infer_target = hash(old_id_hash) % infer_target_per == 0;
      if (is_infer_target) {
        *(infer_target_mask_ptr + i) = 1;
      } else {
        *(infer_target_mask_ptr + i) = 0;
      }
    }
  });
  new_to_old.clear();

  auto new_u = aten::NewIdArray(u.size());
  auto new_u_ptr = (int64_t*) new_u->data;
  auto new_v = aten::NewIdArray(v.size());
  auto new_v_ptr = (int64_t*) new_v->data;

  runtime::parallel_for(0, u.size(), [&](size_t b, size_t e) {
    for (size_t i = b; i < e; i++) {
      *(new_u_ptr + i) = mapping[u[i]];
      *(new_v_ptr + i) = mapping[v[i]];
    }
  });

  auto named_nd_tensors = std::vector<NamedTensor>();
  named_nd_tensors.emplace_back("infer_target_mask", infer_target_mask);
  named_nd_tensors.emplace_back("_ID", dgl_nid);
  named_nd_tensors.emplace_back("new_u", new_u);
  named_nd_tensors.emplace_back("new_v", new_v);
  SaveNdTensors(output_dir + "/result/infer_data.dgl", named_nd_tensors);
}

void BuildPartGraph(int part_id, std::string output_dir, int num_input_edge_partitions, int num_target_parts, int infer_target_per) {
  std::cout << "BuildPartGraph-" << part_id << std::endl;
  auto u = std::vector<int64_t>();
  auto v = std::vector<int64_t>();

  auto new_to_old = std::vector<int64_t>();

  for (int i = 1; i <= num_input_edge_partitions; i++) {
    auto edge_file_path = output_dir + "/edges_" + std::to_string(part_id) + "_" + GetIdxPostfix(i) + ".txt";
    std::ifstream input_stream(edge_file_path);
    int src_id, dst_id;
    while (true) {
      input_stream >> src_id;
      if (src_id < 0) {
        break;
      }
      input_stream >> dst_id;
      u.push_back(src_id);
      v.push_back(dst_id);
    }
  }

  auto mapping = phmap::flat_hash_map<int64_t, int64_t>();
  for (auto& dst_id : v) {
    auto ret = mapping.insert({ dst_id, mapping.size() });
    if (ret.second) {
      new_to_old.push_back(dst_id);
    }
  }

  auto num_nodes_in_this_parititon = new_to_old.size();

  for (auto& src_id : u) {
    auto ret = mapping.insert({ src_id, mapping.size() });
    if (ret.second) {
      new_to_old.push_back(src_id);
    }
  }

  auto num_nodes = mapping.size();
  auto orig_ids = IdArray::FromVector(new_to_old);

  auto part_id_arr = aten::NewIdArray(num_nodes);
  auto part_id_arr_ptr = (int64_t*) part_id_arr->data;
  auto inner_node = aten::NewIdArray(num_nodes, DLContext{kDLCPU, 0}, 8);
  auto inner_node_ptr = (int8_t*) inner_node->data;

  runtime::parallel_for(0, num_nodes, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; i++) {
      auto& old_id = new_to_old[i];
      uint64_t old_id_hash = hash(old_id);
      int64_t pid = old_id_hash % num_target_parts;

      *(part_id_arr_ptr + i) = pid;

      if (i < num_nodes_in_this_parititon) {
        *(inner_node_ptr + i) = 1;
      } else {
        *(inner_node_ptr + i) = 0;
      }
    }
  });
  new_to_old.clear();

  auto new_u = aten::NewIdArray(u.size());
  auto new_u_ptr = (int64_t*) new_u->data;
  auto new_v = aten::NewIdArray(v.size());
  auto new_v_ptr = (int64_t*) new_v->data;

  runtime::parallel_for(0, u.size(), [&](size_t b, size_t e) {
    for (size_t i = b; i < e; i++) {
      *(new_u_ptr + i) = mapping[u[i]];
      *(new_v_ptr + i) = mapping[v[i]];
    }
  });

  auto named_nd_tensors = std::vector<NamedTensor>();
  named_nd_tensors.emplace_back("orig_id", orig_ids);
  named_nd_tensors.emplace_back("inner_node", inner_node);
  named_nd_tensors.emplace_back("part_id", part_id_arr);

  CreateDirectory(output_dir + "/result/part" + std::to_string(part_id));
  SaveNdTensors(output_dir + "/result/part" + std::to_string(part_id) + "/data.dgl", named_nd_tensors);

  named_nd_tensors.clear();
  named_nd_tensors.emplace_back("new_u", new_u);
  named_nd_tensors.emplace_back("new_v", new_v);
  SaveNdTensors(output_dir + "/result/part" + std::to_string(part_id) + "/edges.dgl", named_nd_tensors);
}

void BuildGraphs(std::string base_dir, std::string output_dir, int num_input_edge_partitions, int num_target_parts, int infer_target_per) {
  int max_node_id;
  auto meta_input = std::ifstream(output_dir + "/max_node_id.txt", std::ofstream::in);
  meta_input >> max_node_id;
  std::cout << "max_node_id = " << max_node_id << std::endl;
  meta_input.close();

  CreateDirectory(output_dir + "/result");
  
  omp_set_num_threads(48);

  BuildInferTargetGraph(output_dir, num_input_edge_partitions, num_target_parts, infer_target_per);
  for (int i = 0; i < num_target_parts; i++) {
    BuildPartGraph(i, output_dir, num_input_edge_partitions, num_target_parts, infer_target_per);
  }
}

void Main(int argc, char* argv[]) {
  std::string base_dir = argv[1];
  std::string output_dir = argv[2];
  int num_input_edge_partitions = std::stoi(argv[3]);
  int num_target_parts = std::stoi(argv[4]);
  int infer_target_per = std::stoi(argv[5]); // 1000 means 1/1000 nodes will be inference targets.
  int stage = std::stoi(argv[6]);
  std::cout << "base_dir=" << base_dir << ", output_dir=" << output_dir <<
      ", num_input_edge_partitions=" << num_input_edge_partitions << ", num_target_parts=" << num_target_parts <<
      ", infer_target_per=" << infer_target_per << std::endl;

  if (stage == 0) {
    SplitEdges(base_dir, output_dir, num_input_edge_partitions, num_target_parts, infer_target_per);
  } else if (stage == 1) {
    BuildGraphs(base_dir, output_dir, num_input_edge_partitions, num_target_parts, infer_target_per);
  } else if (stage == -1) {
    SplitEdges(base_dir, output_dir, num_input_edge_partitions, num_target_parts, infer_target_per);
    BuildGraphs(base_dir, output_dir, num_input_edge_partitions, num_target_parts, infer_target_per);
  }
}

}
}


int main(int argc, char* argv[]) {
  dgl::inference::Main(argc, argv);
}
