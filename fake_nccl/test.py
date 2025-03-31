import os
import torch
import torch.distributed as dist

def setup():
    # 从环境变量中获取初始化参数
    # 请确保你在运行前设置了 MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 每个进程使用自己对应的 GPU
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 创建一个 tensor，值为当前进程的 rank
    tensor = torch.ones(1, device=device) * rank

    # 进行 All-Reduce 求和
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 打印各个进程的结果，所有进程的值之和应为 0+1+...+(world_size-1)
    print(f"Rank {rank}/{world_size} has data {tensor.item()}")

    cleanup()

if __name__ == '__main__':
    main()
