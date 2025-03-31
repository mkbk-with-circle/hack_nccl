### 一些命令
gcc -shared -fPIC -o libfake_nccl.so fake_nccl.cc
LD_PRELOAD=/workspace/work/hack_nccl/fake_nccl/libfake_nccl.so python -m torch.distributed.run --nproc_per_node=2 test.py