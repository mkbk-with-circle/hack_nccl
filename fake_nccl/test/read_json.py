import json
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Extract node names from a JSON file.')
parser.add_argument('file_name', type=str, help='Input JSON file name')
args = parser.parse_args()

# 获取输入文件名
file_name = args.file_name

# 构造输出文件名
output_file_name = file_name.replace('.json', '2name.txt')

# 读取并处理 JSON
with open(output_file_name, 'w') as output_file:
    with open(file_name, 'r') as file:
        data = json.load(file)
    for node in data.get("nodes", []):
        node_id = node.get("id")
        node_name = node.get("name")
        output_file.write(f"ID: {node_id}, Name: {node_name}\n")
