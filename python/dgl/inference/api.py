from .._ffi.function import _init_api
from .. import backend as F

def load_tensor(batch_id, name):
    return F.zerocopy_from_dgl_ndarray(_CAPI_DGLInferenceLoadTensor(batch_id, name))

_init_api("dgl.inference.api")
