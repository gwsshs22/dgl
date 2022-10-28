/*!
 *  Copyright (c) 2022 by Contributors
 * \file dgl/inference/launcher.h
 * \brief 
 */
#pragma once

namespace dgl {
namespace inference {

void ExecMasterProcess();

void ExecWorkerProcess();

// Called by a newly spawned process that is managed by the master or the worker process.
void StartActorProcessThread(); 

}  // namespace inference
}  // namespace dgl
