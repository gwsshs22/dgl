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

void WriteTraces(const std::string& breakdown_trace_dir, const std::string& file_name) {
  std::tuple<int, std::string, int> t;

  std::string file_path = breakdown_trace_dir + "/" + file_name + ".txt";
  remove(file_path.c_str());

  std::fstream fs(file_path, std::fstream::out);  
  while (queue.try_dequeue(t)) {
      fs << std::get<0>(t) << "," << std::get<1>(t) << "," << std::get<2>(t) << std::endl;
  }
  fs.close();
}

}
}
