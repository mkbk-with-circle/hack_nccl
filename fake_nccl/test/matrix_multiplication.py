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

            # 模拟一次 NCCL all_reduce（每个进程对计算结果做规约）
            dist.all_reduce(result_on_gpu)

            if epoch == 11:
                et.stop()
            if epoch == 10:
                et.start()
                for i in range(10):
                    dist.all_reduce(result_on_gpu)

            prof.step()

    et.unregister_callback()
    dist.destroy_process_group()
