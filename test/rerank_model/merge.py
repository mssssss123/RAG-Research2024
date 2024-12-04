import os
import json

def merge_list_file(task_lsit, output_file):
    
    merged_data = []
    idx = 0 
    for filename in task_lsit:
        if filename.endswith(".jsonl"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                # 逐行读取 JSONL 文件并添加到 merged_data 列表
                for line in file:
                    json_object = json.loads(line.strip())
                    json_object['id'] = str(idx)
                    # new_jsonl = {}
                    # new_jsonl['id'] = str(idx)
                    # new_jsonl['question'] = json_object['question']
                    # new_jsonl['query'] = json_object['query']
                    # new_jsonl['data_type'] = json_object['data_type']
                    merged_data.append(json_object)
                    idx+=1

    # 将合并的数据写入一个新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in merged_data:
            json.dump(entry, outfile)
            outfile.write('\n')

    print(f"合并完成，输出文件保存为：{output_file}")
    
    
def merge_jsonl_files(input_folder, output_file):
    data_list = []
    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith('.jsonl')]
    jsonl_files.sort()  # 按文件名排序

    for filename in jsonl_files:
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as infile:
                for line in infile:
                    data = json.loads(line)
                    data['_source_file'] = filename
                    data_list.append(data)

    # 将合并的数据写入一个新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in data_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    print(f"合并完成，输出文件保存为：{output_file}")

# 使用示例
input_folder = '/data/groups/QY_LLM_Other/meisen/rerank_test/train_reranktop10/temp/dev'  # 你要合并的 jsonl 文件所在的文件夹
output_file = '/data/groups/QY_LLM_Other/meisen/rerank_test/train_reranktop10/temp/dev/combine.jsonl'
merge_jsonl_files(input_folder, output_file)