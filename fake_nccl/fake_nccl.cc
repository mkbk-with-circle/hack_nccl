#include "include/collectives.h"
#include "include/argcheck.h" // Need some checks here since we access comm
#include "include/enqueue.h"
#include "include/nvtx_payload_schemas.h"
#include "include/core.h"
#include "include/comm.h"
#include "include/group.h"
#include "include/register.h"
#include "include/transport.h"
#include "include/nvtx.h"
#include <iostream>

void printNcclConfig() {
    const char* algo = std::getenv("NCCL_ALGO");
    const char* proto = std::getenv("NCCL_PROTO");

    if (algo) {
        std::cout << "NCCL_ALGO: " << algo << std::endl;
    } else {
        std::cout << "NCCL_ALGO is not set." << std::endl;
    }

    if (proto) {
        std::cout << "NCCL_PROTO: " << proto << std::endl;
    } else {
        std::cout << "NCCL_PROTO is not set." << std::endl;
    }
}
//初始化部分*************************************************
NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist){
    printf("ncclCommInitAll\n");
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
    printf("ncclCommInitRank\n");
    initNvtxRegisteredEnums();
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
    printf("ncclGetUniqueId\n");
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config) {
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommEnsureReady, ncclComm_t comm);
ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count){
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid){
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
ncclResult_t  ncclMemAlloc(void **ptr, size_t size) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclMemFree, void* ptr);
ncclResult_t ncclMemFree(void* ptr) {
    return ncclSuccess;
}
//集合通信部分*************************************************

__global__ void fake_nccl_all_gather(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream){
    nvtxRangePushA("nccl:all_gather");
    printf("nccl:all_gather\n");
    printNcclConfig();
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);

    dim3 threads(1);
    dim3 blocks(1);

    fake_nccl_all_gather<<<blocks, threads, 0, stream>>>(dummy_data);
    // Ensure kernel execution is complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }

    cudaFree(dummy_data);
    nvtxRangePop();
    return ncclSuccess;
}



__global__ void fake_nccl_all_reduce(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}
NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
    //printf("ncclAllReduce\n");
    //NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    //NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));
    //printf("ncclAllReduce done\n");
    nvtxRangePushA("nccl:all_reduce");
    printf("nccl:all_reduce\n");
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);

    dim3 threads(1);
    dim3 blocks(1);

    // ✅ 方式一：推荐
    fake_nccl_all_reduce<<<blocks, threads, 0, stream>>>(dummy_data);

    // ✅ 方式二：你也可以用 cudaLaunchKernel 直接调度
    // void* args[] = { &dummy_data };
    // cudaLaunchKernel((void*)fake_nccl_kernel, blocks, threads, args, 0, stream);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }
    cudaFree(dummy_data);
    nvtxRangePop();
    return ncclSuccess;
}

__global__ void fake_nccl_broadcast(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream){
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);
    printf("nccl:broadcast\n");
    dim3 threads(1);
    dim3 blocks(1);

    fake_nccl_broadcast<<<blocks, threads, 0, stream>>>(dummy_data);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }
    cudaFree(dummy_data);
    return ncclSuccess;
}

__global__ void fake_nccl_bcast(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}

NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);
    printf("nccl:bcast\n");
    dim3 threads(1);
    dim3 blocks(1);

    fake_nccl_bcast<<<blocks, threads, 0, stream>>>(dummy_data);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }
    cudaFree(dummy_data);
  return ncclSuccess;
}

__global__ void fake_nccl_reduce(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}   

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);
    printf("nccl:reduce\n");
    dim3 threads(1);
    dim3 blocks(1);

    fake_nccl_reduce<<<blocks, threads, 0, stream>>>(dummy_data);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }
    cudaFree(dummy_data);
    return ncclSuccess;
}

__global__ void fake_nccl_reduce_scatter(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);
    printf("nccl:reduce_scatter\n");
    dim3 threads(1);
    dim3 blocks(1);

    fake_nccl_reduce_scatter<<<blocks, threads, 0, stream>>>(dummy_data);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }
    cudaFree(dummy_data);
    return ncclSuccess;
}
    
//点对点通信部分*************************************************
__global__ void fake_nccl_send(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);
    printf("nccl:send\n");
    dim3 threads(1);
    dim3 blocks(1);

    fake_nccl_send<<<blocks, threads, 0, stream>>>(dummy_data);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }
    cudaFree(dummy_data);
    return ncclSuccess;
}

__global__ void fake_nccl_recv(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // no-op
    }
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
    float* dummy_data;
    cudaError_t err = cudaMalloc(&dummy_data, sizeof(float) * 1);
    printf("nccl:recv\n");
    dim3 threads(1);
    dim3 blocks(1);

    fake_nccl_recv<<<blocks, threads, 0, stream>>>(dummy_data);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dummy_data);
        return ncclSystemError; // or appropriate error code
    }
    cudaFree(dummy_data);
    return ncclSuccess;
}
//组行为部分*************************************************


NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupSimulateEnd, ncclSimInfo_t* simInfo);
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* simInfo){
    return ncclSuccess;
}


//其他函数*************************************************
NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  return ncclSuccess;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

NCCL_API(ncclResult_t, ncclCommDump, ncclComm_t comm, std::unordered_map<std::string, std::string>& dump);
ncclResult_t ncclCommDump(ncclComm_t comm, std::unordered_map<std::string, std::string>& dump){
    return ncclSuccess;
}
