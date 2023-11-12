#include "trace.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <dmlc/concurrentqueue.h>

namespace dgl {
namespace omega {


typedef dmlc::moodycamel::ConcurrentQueue<std::tuple<int, std::string, int64_t>> QueueT;

QueueT queue;

void PutTrace(int batch_id, const std::string& name, int64_t elapsed_micro) {
  queue.enqueue(std::make_tuple(batch_id, name, elapsed_micro));
}

std::vector<std::tuple<int, std::string, int64_t>> GetCppTraces() {
  std::tuple<int, std::string, int64_t> t;
  std::vector<std::tuple<int, std::string, int64_t>> ret;
  while (queue.try_dequeue(t)) {
    ret.emplace_back(std::move(t));
  }

  return ret;
}

}
}
