# 一些命令
gcc -shared -fPIC -o libfake_nccl.so fake_nccl.cc
nvcc -x cu -Xcompiler -fPIC -shared -o libfake_nccl.so fake_nccl.cc 

LD_PRELOAD=/workspace/work/hack_nccl/fake_nccl/libfake_nccl.so python -m torch.distributed.run --nproc_per_node=2 test.py

LD_PRELOAD=/workspace/work/hack_nccl/fake_nccl/libfake_nccl.so python -m torch.distributed.run --nproc_per_node=2 matrix_multiplication.py

python -m torch.distributed.run --nproc_per_node=2 matrix_multiplication.py
# 添加你自己的头文件目录到CPATH
export CPATH=/home/ymy/workspace/hack_nccl/fake_nccl/include:\
/home/ymy/workspace/hack_nccl/fake_nccl/include/plugin:\
/home/ymy/workspace/hack_nccl/fake_nccl/include/nvtx3:\
/usr/local/cuda/include:\
$CPATH

export CPATH=/workspace/work/hack_nccl/fake_nccl/include:\
/workspace/work/hack_nccl/fake_nccl/include/plugin:\
/workspace/work/hack_nccl/fake_nccl/include/nvtx3:\
/usr/local/cuda/include:\
$CPATH


/workspace/work/hack_nccl/fake_nccl/test/pre_pytorch_et_rank0.json
/workspace/work/hack_nccl/fake_nccl/test/pre_kineto_trace_rank0.json
/workspace/work/hack_nccl/fake_nccl/test/pre_chakra.json

**运行chakra的代码**
$ cd ~/
$ python -m venv venv
$ source venv/bin/activate
$ pip install numpy torch
$ python matmul.py



chakra_trace_link --rank 2 --chakra-host-trace /workspace/work/hack_nccl/fake_nccl/test/pre_pytorch_et_rank0.json --chakra-device-trace /workspace/work/hack_nccl/fake_nccl/test/pre_kineto_trace_rank0.json --output-file /workspace/work/hack_nccl/fake_nccl/test/pre_chakra.json




chakra_trace_link --rank 2 --chakra-host-trace /workspace/work/hack_nccl/fake_nccl/test/pytorch_et_rank0.json --chakra-device-trace /workspace/work/hack_nccl/fake_nccl/test/kineto_trace_rank0.json --output-file /workspace/work/hack_nccl/fake_nccl/test/hack_chakra.json
