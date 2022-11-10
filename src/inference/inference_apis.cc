/*!
 *  Copyright (c) 2022 by Contributors
 * \file inference/inference_apis.cc
 * \brief DGL inference APIs
 */
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"

#include "execution/mem_utils.h"
#include "process/object_storage.h"
#include "entrypoint.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;

namespace dgl {

struct ActorRequestWrapper : public runtime::Object {
  uint64_t req_id;
  int request_type;
  int batch_id;

  ActorRequestWrapper(uint64_t rid, int rtype, int bid)
      : req_id(rid), request_type(rtype), batch_id(bid) {
  }

  static constexpr const char* _type_key = "inference.process.ActorRequest";
  DGL_DECLARE_OBJECT_TYPE_INFO(ActorRequestWrapper, runtime::Object);    
};

DGL_DEFINE_OBJECT_REF(ActorRequestWrapperRef, ActorRequestWrapper);


DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceExecMasterProcess")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    inference::ExecMasterProcess();
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceExecWorkerProcess")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    inference::ExecWorkerProcess();
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceStartActorProcessThread")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    inference::StartActorProcessThread();
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceActorNotifyInitialized")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    inference::ActorNotifyInitialized();
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceActorFetchRequest")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    auto actor_request = inference::ActorFetchRequest();
    auto wrapper = std::make_shared<ActorRequestWrapper>(actor_request.req_id(), actor_request.request_type(), actor_request.batch_id());
    *rv = wrapper;
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceActorRequestGetRequestType")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    const ActorRequestWrapperRef wrapper = args[0];
    *rv = wrapper->request_type;
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceActorRequestGetBatchId")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    const ActorRequestWrapperRef wrapper = args[0];
    *rv = wrapper->batch_id;
  });

DGL_REGISTER_GLOBAL("inference.process._CAPI_DGLInferenceActorRequestDone")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    const ActorRequestWrapperRef wrapper = args[0];
    inference::ActorRequestDone(wrapper->req_id);
  });

DGL_REGISTER_GLOBAL("inference.api._CAPI_DGLInferenceLoadTensor")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int batch_id = args[0];
    std::string name = args[1];

    *rv = inference::LoadFromSharedMemory(batch_id, name);
  });

DGL_REGISTER_GLOBAL("inference.api._CAPI_DGLInferencePutTensor")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int batch_id = args[0];
    std::string name = args[1];
    NDArray src_arr = args[2];

    inference::ObjectStorage::GetInstance()->CopyToSharedMemory(batch_id, name, src_arr);
  });

DGL_REGISTER_GLOBAL("inference.api._CAPI_DGLInferenceTensorCleanup")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    int batch_id = args[0];
    inference::ObjectStorage::GetInstance()->Cleanup(batch_id);
  });

}  // namespace dgl
