import json
output_file_path = '/home/meis23/project/FiD-main/open_domain_data/NQ/new_train.json'

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
data = read_jsonl('/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/train_LLM_data/minicpm_top5/marco_LLM_rerank/sft_dev.jsonl')
# 读取 JSON 文件
# with open('/data/groups/QY_LLM_Other/meisen/radit/dataset/sft/train_noread.jsonl', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# for example in data:
#     ctxs = example['ctxs']
#     for p in ctxs:
#         if 'score' in p:
#             print('---')

# for example in data:
#     example['ctxs'] = example['ctxs'][:20]

# with open(output_file_path, 'w', encoding='utf-8') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)
print("-------")