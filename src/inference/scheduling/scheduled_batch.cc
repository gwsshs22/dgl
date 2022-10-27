#include "scheduled_batch.h"

namespace dgl {
namespace inference {

ScheduledBatch::ScheduledBatch(int batch_id, BatchInput batch_input)
    : batch_id_(batch_id), batch_input_(std::move(batch_input)), status_(kCreated) {
}

void ScheduledBatch::SetStatus(const Status& to) {
  status_ = to;
}

}
}
