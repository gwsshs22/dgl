#include <dgl/inference/envs.h>
#include <dgl/inference/common.h>

#include <fstream>
#include <mutex>
#include <cstdio>

namespace dgl {
namespace inference {

const char* DGL_INFER_MASTER_HOST = "DGL_INFER_MASTER_HOST";
const char* DGL_INFER_MASTER_PORT = "DGL_INFER_MASTER_PORT";
const char* DGL_INFER_NODE_RANK = "DGL_INFER_NODE_RANK";
const char* DGL_INFER_LOCAL_RANK = "DGL_INFER_LOCAL_RANK";
const char* DGL_INFER_NUM_NODES = "DGL_INFER_NUM_NODES";
const char* DGL_INFER_NUM_BACKUP_SERVERS = "DGL_INFER_NUM_BACKUP_SERVERS";
const char* DGL_INFER_NUM_DEVICES_PER_NODE = "DGL_INFER_NUM_DEVICES_PER_NODE";
const char* DGL_INFER_NUM_SAMPLERS_PER_NODE = "DGL_INFER_NUM_SAMPLERS_PER_NODE";
const char* DGL_INFER_IFACE = "DGL_INFER_IFACE";
const char* DGL_INFER_ACTOR_PROCESS_ROLE = "DGL_INFER_ACTOR_PROCESS_ROLE";
const char* DGL_INFER_ACTOR_PROCESS_GLOBAL_ID = "DGL_INFER_ACTOR_PROCESS_GLOBAL_ID";

const char* DGL_INFER_PARALLELIZATION_TYPE = "DGL_INFER_PARALLELIZATION_TYPE";
const char* DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS = "DGL_INFER_USING_PRECOMPUTED_AGGREGATIONS";

const char* DGL_INFER_INPUT_TRACE_DIR = "DGL_INFER_INPUT_TRACE_DIR";
const char* DGL_INFER_NUM_WARMUPS = "DGL_INFER_NUM_WARMUPS";
const char* DGL_INFER_NUM_REQUESTS = "DGL_INFER_NUM_REQUESTS";
const char* DGL_INFER_RESULT_DIR = "DGL_INFER_RESULT_DIR";
const char* DGL_INFER_COLLECT_STATS = "DGL_INFER_COLLECT_STATS";
const char* DGL_INFER_EXECUTE_ONE_BY_ONE = "DGL_INFER_EXECUTE_ONE_BY_ONE";

bool TRACE_ENABLED = false;

void EnableTracing() {
  TRACE_ENABLED = true;
}

class TraceCollector {

 public:
  inline void Add(const TraceMe& trace_me) {
    std::lock_guard<std::mutex> guard(mtx_);
    traces_.push_back(std::make_tuple(trace_me.batch_id(), trace_me.name(), trace_me.GetElapsedMicro()));
  }

  void Write(const std::string result_dir, int node_rank) {
    std::lock_guard<std::mutex> guard(mtx_);
    std::string file_path = result_dir + "/node_" + std::to_string(node_rank) + ".txt";
    remove(file_path.c_str());
    std::fstream fs(file_path, std::fstream::out);
    for (const auto& t : traces_) {
      fs << std::get<0>(t) << "," << std::get<1>(t) << "," << std::get<2>(t) << std::endl;
    }
    fs.close();
  }

 private:
  std::vector<std::tuple<int, const char*, int>> traces_;
  std::mutex mtx_;
};

static TraceCollector trace_collector;

void AddTrace(const TraceMe& trace_me) {
  trace_collector.Add(trace_me);
}

void WriteTraces(const std::string result_dir, int node_rank) {
  trace_collector.Write(result_dir, node_rank);
}

}
}
