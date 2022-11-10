from .._ffi.function import _init_api
from .. import backend as F

def load_tensor(batch_id, name):
    return F.zerocopy_from_dgl_ndarray(_CAPI_DGLInferenceLoadTensor(batch_id, name))

def put_tensor(batch_id, name, tensor):
    _CAPI_DGLInferencePutTensor(batch_id, name, F.zerocopy_to_dgl_ndarray(tensor))

_init_api("dgl.inference.api")
