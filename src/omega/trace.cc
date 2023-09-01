#include "trace.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <dmlc/concurrentqueue.h>

namespace dgl {
namespace omega {


typedef dmlc::moodycamel::ConcurrentQueue<std::tuple<int, std::string, int>> QueueT;

QueueT queue;

void PutTrace(int batch_id, const std::string& name, int elapsed_micro) {
  queue.enqueue(std::make_tuple(batch_id, name, elapsed_micro));
}

std::vector<std::tuple<int, std::string, int>> GetCppTraces() {
  std::tuple<int, std::string, int> t;
  std::vector<std::tuple<int, std::string, int>> ret;
  while (queue.try_dequeue(t)) {
    ret.emplace_back(std::move(t));
  }

  return ret;
}

}
}
