

import json
import os


root_data_path = '/data/groups/QY_LLM_Other/meisen/radit/dataset/dpo/train_noread'


def get_files(path):
    # 获取指定路径下的所有项目
    all_items = os.listdir(path)
    
    # 筛选出所有文件
    files = [item for item in all_items if os.path.isfile(os.path.join(path, item))]
    
    return files

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_to_jsonl(data, file_path):    
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            json_record = json.dumps(record, ensure_ascii=False)
            f.write(json_record + '\n')

all_data = []
file_name = get_files(root_data_path)
for file in file_name:
    cur_file_path = os.path.join(root_data_path, file)
    data = read_jsonl(cur_file_path)
    all_data.extend(data)


save_to_jsonl(all_data,'/data/groups/QY_LLM_Other/meisen/radit/dataset/dpo/train_noread.jsonl')
print('finish')