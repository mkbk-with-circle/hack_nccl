#include "include/nccl.h"
#include "include/collectives.h"
#include "include/argcheck.h" // Need some checks here since we access comm
#include "include/enqueue.h"
#include "include/nccl.h"
#include "include/nvtx_payload_schemas.h"
//初始化部分*************************************************
NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank){
    return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream){
    return ncclSuccess;
}

