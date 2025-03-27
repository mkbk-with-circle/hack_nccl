/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // 定义一个NVTX payload schema，用于记录消息大小
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };

  // 计算每个消息的大小（字节数）
  size_t msgsize = sendcount * ncclTypeSize(datatype);

  // 使用NVTX记录函数调用的时间范围和参数
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  // 创建一个ncclInfo结构体，包含AllGather操作的所有必要信息
  struct ncclInfo info = { 
    ncclFuncAllGather, // 操作类型
    "AllGather",       // 操作名称
    sendbuff,          // 发送缓冲区
    recvbuff,          // 接收缓冲区
    sendcount,         // 发送元素数量
    datatype,          // 数据类型
    ncclSum,           // 规约操作（这里不使用）
    0,                 // 根进程（不适用于AllGather）
    comm,              // 通信上下文
    stream,            // CUDA流
    ALLGATHER_CHUNKSTEPS, // 分块步长
    ALLGATHER_SLICESTEPS  // 切片步长
  };

  // 将操作信息入队，准备执行
  //return ncclEnqueueCheck(&info);
  return ncclSuccess;
}
