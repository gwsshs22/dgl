/*!
 *  Copyright (c) 2022 by Contributors
 * \file inference/inference_apis.cc
 * \brief DGL inference APIs
 */
#include <dgl/array.h>

#include "./process/process.h"
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;

namespace dgl {

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceExecMasterProcess")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    inference::ExecMasterProcess();
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceExecWorkerProcess")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    inference::ExecWorkerProcess();
  });

}  // namespace dgl
