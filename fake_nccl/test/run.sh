# 参数有：
# 1. --node 一个整数，表示节点数
# 2. --if_link 是否链接动态库，1表示链接，0表示不链接
# 3. --name 一个字符串，用来命名，若有该参数则把这个字符串作为前缀加到所有的生成的文件中，若无则不进行操作
# 4. --chakra_rank 一个整数，表示chakra的rank


# 先执行：
# cd ..
# nvcc -x cu -Xcompiler -fPIC -shared -o libfake_nccl.so fake_nccl.cc 
# cd test

# 分支判断if_link是否为0
# 若不链接则： python -m torch.distributed.run --nproc_per_node=node matrix_multiplication.py
# 若链接则： LD_PRELOAD=/workspace/work/hack_nccl/fake_nccl/libfake_nccl.so python -m torch.distributed.run --nproc_per_node=node matrix_multiplication.py

# 若有name，则遍历从kineto_trace_rank0.json到kineto_trace_rank{node-1}.json，依次替换前缀，并重命名为{name}_kineto_trace_rank{i}.json
# 若有name，则遍历从pytorch_et_rank0.json到pytorch_et_rank{node-1}.json，依次替换前缀，并重命名为{name}_pytorch_et_rank{i}.json


# 最后若有chakra_rank参数，则运行:
# 若有name参数则：chakra_trace_link --rank {node} --chakra-host-trace {name}_pytorch_et_rank{chakra_rank}.json --chakra-device-trace {name}_kineto_trace_rank{chakra_rank}.json --output-file {name}_chakra.json
# 若无name参数则：chakra_trace_link --rank {node} --chakra-host-trace pytorch_et_rank{chakra_rank}.json --chakra-device-trace kineto_trace_rank{chakra_rank}.json --output-file chakra.json

#!/bin/bash

#./run.sh --node 4 --if_link 1 --name test1 --chakra_rank 2


# 默认参数初始化
node=1
if_link=0
name=""
chakra_rank=-1

# 参数解析
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --node)
      node="$2"
      shift; shift
      ;;
    --if_link)
      if_link="$2"
      shift; shift
      ;;
    --name)
      name="$2"
      shift; shift
      ;;
    --chakra_rank)
      chakra_rank="$2"
      shift; shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# 编译 fake_nccl 动态库
cd ..
nvcc -x cu -Xcompiler -fPIC -shared -o libfake_nccl.so fake_nccl.cc
cd test

# 运行分布式程序
if [ "$if_link" -eq 0 ]; then
  python -m torch.distributed.run --nproc_per_node=$node matrix_multiplication.py
else
  LD_PRELOAD=/workspace/work/hack_nccl/fake_nccl/libfake_nccl.so \
  python -m torch.distributed.run --nproc_per_node=$node matrix_multiplication.py
fi

# 重命名 trace 文件
if [ -n "$name" ]; then
  for ((i=0; i<node; i++)); do
    if [ -f "kineto_trace_rank${i}.json" ]; then
      mv "kineto_trace_rank${i}.json" "${name}_kineto_trace_rank${i}.json"
    fi
    if [ -f "pytorch_et_rank${i}.json" ]; then
      mv "pytorch_et_rank${i}.json" "${name}_pytorch_et_rank${i}.json"
    fi
  done
fi

# 生成 chakra trace
if [ "$chakra_rank" -ge 0 ]; then
  if [ -n "$name" ]; then
    chakra_trace_link --rank ${node} \
      --chakra-host-trace "${name}_pytorch_et_rank${chakra_rank}.json" \
      --chakra-device-trace "${name}_kineto_trace_rank${chakra_rank}.json" \
      --output-file "${name}_chakra.json"
  else
    chakra_trace_link --rank ${node} \
      --chakra-host-trace "pytorch_et_rank${chakra_rank}.json" \
      --chakra-device-trace "kineto_trace_rank${chakra_rank}.json" \
      --output-file "chakra.json"
  fi
fi

# 提取节点 ID 和 Name
if [ "$chakra_rank" -ge 0 ]; then
  if [ -n "$name" ]; then
    python read_json.py "${name}_chakra.json"
  else
    python read_json.py "chakra.json"
  fi
fi
