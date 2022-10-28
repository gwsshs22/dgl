/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/inference/launcher.h
 * \brief 
 */
#pragma once

#include "./process/master_process.h"
#include "./process/worker_process.h"
#include "./process/actor_process.h"

namespace dgl {
namespace inference {

void ExecMasterProcess();

void ExecWorkerProcess();

// Called by a newly spawned process that is managed by the master or the worker process.
void StartActorProcessThread();

}  // namespace inference
}  // namespace dgl
