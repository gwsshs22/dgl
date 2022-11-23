#pragma once

#include <dlpack/dlpack.h>
#include <dmlc/memory_io.h>

#include <dgl/inference/common.h>

namespace dgl {
namespace inference {

inline std::string GetArrayDataName(int batch_id, const std::string& name) {
  return  "/d-" + std::to_string(batch_id) + "_" + name;
}

inline std::string GetArrayMetadataName(int batch_id, const std::string& name) {
  return  "/m-" + std::to_string(batch_id) + "_" + name;
}

static constexpr unsigned int __METADATA_MAX_BYTES = 512;

inline NDArray CreateEmptyArrayFromMetadata(dmlc::Stream& stream) {
  int ndim;
  int64_t *shape;
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
  DLDeviceType device_type;
  int device_id;

  stream.Read(&ndim);
  shape = new int64_t[ndim];
  stream.ReadArray(shape, ndim);
  stream.Read(&code);
  stream.Read(&bits);
  stream.Read(&lanes);
  stream.Read(&device_type);
  stream.Read(&device_id);

  auto shape_vector = std::vector<int64_t>();
  for (auto itr = shape; itr != shape + ndim; itr++) {
    shape_vector.push_back(*itr);
  }

  delete shape;

  return NDArray::Empty(shape_vector,
      DLDataType{code, bits, lanes},
      DLContext{device_type, device_id});
}

inline void WriteMetadata(dmlc::Stream& stream,
                          const NDArray& data) {
  stream.Write(data->ndim);
  stream.WriteArray(data->shape, data->ndim);
  stream.Write(data->dtype.code);
  stream.Write(data->dtype.bits);
  stream.Write(data->dtype.lanes);
  stream.Write(data->ctx.device_type);
  stream.Write(data->ctx.device_id);
}

inline std::shared_ptr<runtime::SharedMemory> CreateMetadataSharedMem(const std::string& name,
                                                                      const NDArray& data) {
  auto metadata = std::make_shared<runtime::SharedMemory>(name);
  auto buf = metadata->CreateNew(__METADATA_MAX_BYTES);

  dmlc::MemoryFixedSizeStream strm(buf, __METADATA_MAX_BYTES);
  WriteMetadata(strm, data);
  return metadata;
}

inline std::shared_ptr<runtime::SharedMemory> CreateMetadata(int batch_id, const std::string& name, const NDArray& arr) {
  return CreateMetadataSharedMem(GetArrayMetadataName(batch_id, name), arr);
}

inline NDArray CopyToSharedMem(int batch_id, const std::string& name, const NDArray& src_arr) {
  if (src_arr.NumElements() == 0) {
    return src_arr;
  }

  NDArray copied = NDArray::EmptySharedUnmanaged(
      GetArrayDataName(batch_id, name),
      std::vector<int64_t>(src_arr->shape, src_arr->shape + src_arr->ndim),
      src_arr->dtype,
      DLContext{kDLCPU, 0},
      true);

  copied.CopyFrom(src_arr);

  return copied;
}

// It turns out that it is impossible to share device memory between processes.
// This will not be called.
inline NDArray CopyToGpu(const NDArray& src_arr, int local_rank) {
  NDArray copied = NDArray::Empty(
      std::vector<int64_t>(src_arr->shape, src_arr->shape + src_arr->ndim),
      src_arr->dtype,
      DLContext{kDLGPU, local_rank});

  copied.CopyFrom(src_arr);

  return copied;
}

inline NDArray LoadFromSharedMemory(int batch_id, const std::string& name) {
  auto metadata = runtime::SharedMemory(GetArrayMetadataName(batch_id, name));
  auto metadata_buf = metadata.Open(__METADATA_MAX_BYTES);

  dmlc::MemoryFixedSizeStream strm(metadata_buf, __METADATA_MAX_BYTES);

  int ndim;
  int64_t *shape;
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
  DLDeviceType device_type;
  int device_id;

  shape = new int64_t[ndim];

  {
    dmlc::Stream& stream = strm;
    stream.Read(&ndim);    
    stream.ReadArray(shape, ndim);
    stream.Read(&code);
    stream.Read(&bits);
    stream.Read(&lanes);
    stream.Read(&device_type);
    stream.Read(&device_id);
  }

  int64_t num_elems = 1;
  auto shape_vector = std::vector<int64_t>();
  for (auto itr = shape; itr != shape + ndim; itr++) {
    num_elems *= *itr;
    shape_vector.push_back(*itr);
  }

  delete shape;

  if (device_type == kDLCPU) {
    if (num_elems == 0) {
      return NDArray::Empty(shape_vector, DLDataType{code, bits, lanes},  DLContext{kDLCPU, 0});
    } else {
      return NDArray::EmptyShared(GetArrayDataName(batch_id, name), shape_vector, DLDataType{code, bits, lanes},  DLContext{kDLCPU, 0}, false);
    }
  } else {
    // It turns out that it is impossible to share device memory between processes.
    CHECK(false);
    assert(device_type == kDLGPU);
    NDArray ret;
    return ret;
  }
}

}
}
