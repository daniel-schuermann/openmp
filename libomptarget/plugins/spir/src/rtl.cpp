//===----RTLs/spir/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// RTL for SPIR-V/OpenCL machine
//
//===----------------------------------------------------------------------===//


#include <cassert>
#include <cstring>
#include <cstdlib>
#include <list>
#include <vector>
#include <gelf.h>
#include <CL/cl.h>

#include "omptargetplugin.h"

#ifndef TARGET_NAME
#define TARGET_NAME SPIR
#endif

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)
#define DP(...) DEBUGP("Target " GETNAME(TARGET_NAME) " RTL", __VA_ARGS__)

#define NUMBER_OF_DEVICES 1
#define OFFLOADSECTIONNAME ".omp_offloading.entries"

#pragma OPENCL EXTENSION cl_khr_spir : enable

/// Keep entries table per device.
struct FuncOrGblEntryTy {
    __tgt_target_table Table;
    std::vector<__tgt_offload_entry> Entries;
    std::vector<cl_kernel> Kernels;
};

/// Class containing all the device information.
class RTLDeviceInfoTy {

public:
    cl_uint numDevices;
    cl_platform_id platformId;
    // per device information
    std::vector<cl_device_id> deviceIDs;
    std::vector<int32_t> maxWorkGroups;
    std::vector<int32_t> nativeVectorWidth;
    std::vector<cl_context> CTX;
    std::vector<cl_command_queue> Queues;
    std::vector<FuncOrGblEntryTy> FuncGblEntries;

    RTLDeviceInfoTy() {
      DP("Start initializing OpenCL\n");
      // get available platforms
      cl_uint platformIdCount = 0;
      clGetPlatformIDs(0, nullptr, &platformIdCount);
      std::vector<cl_platform_id> platformIds(platformIdCount);
      clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

      // check version and devices
      for(cl_platform_id id : platformIds) {
        char buffer[128];
        clGetPlatformInfo(id, CL_PLATFORM_VERSION, 128, buffer, NULL);
        if(strncmp("OpenCL 2.0 AMD", buffer, 14)) {
          continue;
        }
        DP("cl platform version is %s\n", buffer);

        clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        deviceIDs.resize(numDevices);
        clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, numDevices, deviceIDs.data(), nullptr);
        DP("Found %d OpenCL devices\n", numDevices);

        maxWorkGroups.resize(numDevices);
        nativeVectorWidth.resize(numDevices);
        CTX.resize(numDevices);
        Queues.resize(numDevices);
        FuncGblEntries.resize(numDevices);
        platformId = id;

        // get device specific information
        for(int i = 0; i < numDevices; i++) {
          cl_device_id deviceId = deviceIDs[i];
          clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &maxWorkGroups[i], nullptr);
          clGetDeviceInfo(deviceId, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, 4, &nativeVectorWidth[i], nullptr);
          size_t size;
          clGetDeviceInfo(deviceId, CL_DEVICE_MAX_PARAMETER_SIZE, 8, &size, nullptr);
          DP("max parameter size is %zd\n", size);
          cl_ulong memsize;
          clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &memsize, nullptr);
          DP("global mem size is %zd\n", memsize);

          DP("max WGs is: %d\n", maxWorkGroups[i]);
          DP("vector width is: %d\n", nativeVectorWidth[i]);
          cl_uint addressmode;
          clGetDeviceInfo(deviceId, CL_DEVICE_ADDRESS_BITS, 4, &addressmode, nullptr);
          DP("addressing mode is %d bit\n", addressmode);
        }
        //return;
      }
      //numDevices = 0;
      //DP("No OpenCL devices found.\n");
    }
};

static RTLDeviceInfoTy DeviceInfo;

#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
  return true; //TODO
}

int32_t __tgt_rtl_number_of_devices() { return DeviceInfo.numDevices; } //fixme

int32_t __tgt_rtl_init_device(int32_t device_id) {
  cl_int status;

  DP("Initialize SPIR device\n");

  // create context
  cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) DeviceInfo.platformId, 0};
  DeviceInfo.CTX[device_id] = clCreateContext(props, 1, &DeviceInfo.deviceIDs[device_id], nullptr, nullptr, &status);

  if(status != CL_SUCCESS) {
    DP("OpenCL Error: Failed to create context: %d\n", status);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {

  DP("Dev %d: load binary from " DPxMOD " image\n", device_id,
     DPxPTR(image->ImageStart));

  assert(device_id >= 0 && device_id < NUMBER_OF_DEVICES && "bad dev id");

  size_t ImageSize = (size_t)image->ImageEnd - (size_t)image->ImageStart;
  size_t NumEntries = (size_t)(image->EntriesEnd - image->EntriesBegin);
  DP("Expecting to have %zd entries defined.\n", NumEntries);

  /*
  // For debugging purposes, we can write out the spir binary
  char tmp_name[] = "/tmp/tmpfile_XXXXXX";
  int tmp_fd = mkstemp(tmp_name);

  if (tmp_fd == -1) {
    return NULL;
  }

  FILE *ftmp = fdopen(tmp_fd, "wb");

  if (!ftmp) {
    return NULL;
  }

  fwrite(image->ImageStart, ImageSize, 1, ftmp);
  fclose(ftmp);
  DP("written to tmp\n")
  */
  // create CommandQueue and Program
  cl_int status;
  DeviceInfo.Queues[device_id] = clCreateCommandQueueWithProperties(
          DeviceInfo.CTX[device_id], DeviceInfo.deviceIDs[device_id], nullptr, &status);
  if(status != 0) {
    DP("OpenCL Error: Failed to create CommandQueue: %d\n", status);
    return NULL;
  } // should we create the command queue here or on kernel execution or on initialization?
  cl_program program = clCreateProgramWithIL(DeviceInfo.CTX[device_id], image->ImageStart, ImageSize, &status);
  if(status != 0) {
    DP("OpenCL Error: Failed to create program: %d\n", status);
    return NULL;
  }

  status = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
  if(status != 0) {
    DP("OpenCL Error: Failed to build program: %d\n", status);
    return NULL;
  } else {
    DP("OpenCL: Successfully build program\n");
  }

  // create kernel and target entries
  DeviceInfo.FuncGblEntries[device_id].Entries.resize(NumEntries);
  DeviceInfo.FuncGblEntries[device_id].Kernels.resize(NumEntries);
  std::vector<__tgt_offload_entry>& entries = DeviceInfo.FuncGblEntries[device_id].Entries;
  std::vector<cl_kernel>& kernels = DeviceInfo.FuncGblEntries[device_id].Kernels;
  for(int i = 0; i< NumEntries; i++) {
    char *name = image->EntriesBegin[i].name;
    kernels[i] = clCreateKernel(program, name, &status);
    if (status != 0) {
      DP("OpenCL Error: Failed to create kernel %s, %d\n", name, status);
      return NULL;
    }
    entries[i].addr = &kernels[i];
    entries[i].name = name;
  }

  __tgt_target_table &table = DeviceInfo.FuncGblEntries[device_id].Table;
  table.EntriesBegin = &(entries[0]);
  table.EntriesEnd = &(entries[entries.size()]);
  return &table;
}

void *__tgt_rtl_data_alloc(int32_t device_id, int64_t size, void *hst_ptr) {
  cl_int status=0;
  cl_mem mem = clCreateBuffer(DeviceInfo.CTX[device_id], CL_MEM_READ_WRITE, size, NULL, &status);
  if(status != CL_SUCCESS) {
    DP("OpenCL Error: Failed to allocate memory: %d\n", status);
    return NULL;
  }
  return mem;
}

int32_t __tgt_rtl_data_submit(int32_t device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  cl_int status = clEnqueueWriteBuffer(DeviceInfo.Queues[device_id], (cl_mem) tgt_ptr, CL_TRUE, 0, size, hst_ptr, 0,
                                       nullptr, nullptr);

  if(status != CL_SUCCESS) {
    DP("OpenCL Error: Failed to write buffer: %d\n", status);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int32_t device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  cl_int status = clEnqueueReadBuffer(DeviceInfo.Queues[device_id], (cl_mem) tgt_ptr, CL_TRUE, 0, size, hst_ptr, 0,
                                       nullptr, nullptr);
  if(status != CL_SUCCESS) {
    DP("OpenCL Error: Failed to read buffer: %d\n", status);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_delete(int32_t device_id, void *tgt_ptr) {
  if(cl_int status = clReleaseMemObject((cl_mem) tgt_ptr) != CL_SUCCESS) {
    DP("OpenCL Error: Failed to release buffer: %d\n", status);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args, ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t team_num,
                                         int32_t thread_limit, uint64_t loop_tripcount /*not used*/) {
  cl_int status;
  DP("device id is %d\n", device_id);
  // debug...
  cl_kernel * kernel = static_cast<cl_kernel *> (tgt_entry_ptr);
  char buffer[128];
  clGetKernelInfo(*kernel, CL_KERNEL_FUNCTION_NAME, 128, buffer, nullptr);
  cl_uint n;
  clGetKernelInfo(*kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &n, nullptr);
  DP("number of kernel parameters: %d\n", n);
  DP("number of arguments: %d\n", arg_num);


  // set kernel args
  std::vector<void *> ptrs(arg_num-1);
  for (int32_t i = 0; i < arg_num -1 /* ignore omp handle */; ++i) {

    ptrs[i] = (void *)((intptr_t)tgt_args[i] + tgt_offsets[i]);
    status = clSetKernelArg(*kernel, i, /*sizeof(intptr_t) + tgt_offsets[i]*/ sizeof(cl_mem), &ptrs[i]);
    if(status != CL_SUCCESS) {
      DP("OpenCL Error: Failed to set kernel arg %d for %s: %d\n", i, buffer, status);
      return OFFLOAD_FAIL;
    } else {
      DP("OpenCL: Kernel Arg %d set successfully\n", i);
    }
  }

  // calculate number of threads in each team:
  size_t local_work_size;
  clGetKernelWorkGroupInfo(*kernel, DeviceInfo.deviceIDs[device_id], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                           sizeof(size_t), &local_work_size, nullptr);
  DP("preferred workgroup size multiple is: %zd\n", local_work_size);
  if(thread_limit && local_work_size > thread_limit) { // thread limit == 1 means unspecified for whatever reason
    // we should not create teams bigger than thread_limit
    local_work_size = thread_limit;
  }

  // calculate total number of threads to execute the kernel:
  size_t global_work_size;
  if(team_num) { // num_teams() is specified
    global_work_size = local_work_size * team_num;
  } else {
    global_work_size = local_work_size * DeviceInfo.maxWorkGroups[device_id] * 8; // have sane defaults
  }

  // run kernel:
  DP("thread limit is %d, team num is %d\n", thread_limit, team_num);
  DP("global work size is %zd\n", global_work_size);
  DP("local work size is %zd\n", local_work_size);
  status = clEnqueueNDRangeKernel(
          DeviceInfo.Queues[device_id], *kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
  if(status != CL_SUCCESS) {
    DP("OpenCL Error: Failed to enqueue kernel: %d\n", status);
    return OFFLOAD_FAIL;
  }

  DP("OpenCL: Started executing kernel.\n");
  status = clFinish(DeviceInfo.Queues[device_id]);
  if(status != CL_SUCCESS) {
    DP("OpenCL Error: Failed to execute kernel: %d\n", status);
  } else {
    DP("OpenCL: Successfully finished kernel execution.\n")
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets, int32_t arg_num) {
  // use one team!
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, 1, 0, 0);
}

#ifdef __cplusplus
}
#endif