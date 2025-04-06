import os
import torch
import numpy as np
import torch.distributed as dist
from torch.profiler import ExecutionTraceObserver, profile


def trace_handler(prof):
    prof.export_chrome_trace(f"kineto_trace_rank{dist.get_rank()}.json")


def gpu_matrix_multiplication(matrix1: np.ndarray, matrix2: np.ndarray) -> torch.Tensor:
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError("Matrices have incompatible shapes for multiplication.")

    matrix1_torch = torch.tensor(matrix1, dtype=torch.float).cuda()
    matrix2_torch = torch.tensor(matrix2, dtype=torch.float).cuda()

    result_gpu = torch.matmul(matrix1_torch, matrix2_torch)

    return result_gpu


def init_process_group():
    # 通常用于 NCCL 后端初始化（必须设置环境变量）
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


if __name__ == "__main__":
    # 初始化 NCCL 通信组
    init_process_group()

    et = ExecutionTraceObserver()
    et.register_callback(f"pytorch_et_rank{dist.get_rank()}.json")

    # 定义随机矩阵
    matrix_a = np.random.rand(1024, 1024)
    matrix_b = np.random.rand(1024, 1024)

    with profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=10, active=1),
        on_trace_ready=trace_handler,
        record_shapes=True
    ) as prof:
            for epoch in range(20):
                result_on_gpu = gpu_matrix_multiplication(matrix_a, matrix_b)


                # 开始记录 ExecutionTrace（第10轮开始）
                if epoch == 10:
                    et.start()
                    dist.all_reduce(result_on_gpu)
                    # 模拟一次 all_reduce（多次）
                    dist.all_reduce(result_on_gpu)

                    # 模拟 broadcast（从 rank 0 广播给其他）
                    dist.broadcast(result_on_gpu, src=0)

                    # 模拟 reduce（从各 rank 收集到 rank 0）
                    dist.reduce(result_on_gpu, dst=0)

                    # 模拟 reduce_scatter（把结果均分发给所有 rank）
                    recv_buffer = torch.empty_like(result_on_gpu)
                    dist.reduce_scatter(recv_buffer, list(result_on_gpu.chunk(dist.get_world_size())))

                    # 模拟 all_gather（每个 rank 都收集所有 rank 的数据）
                    gathered = [torch.empty_like(result_on_gpu) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered, result_on_gpu)

                    # 模拟 send/recv（rank 0 发送给 1，1 接收）
                    if dist.get_world_size() >= 2:
                        tag = 123
                        if dist.get_rank() == 0:
                            dist.send(result_on_gpu, dst=1, tag=tag)
                        elif dist.get_rank() == 1:
                            recv = torch.empty_like(result_on_gpu)
                            dist.recv(recv, src=0, tag=tag)


                # 停止记录 ExecutionTrace（第11轮）
                if epoch == 11:
                    et.stop()

                prof.step()


    et.unregister_callback()
    dist.destroy_process_group()
