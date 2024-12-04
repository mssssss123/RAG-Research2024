import json

data_file_path = '/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/train_LLM_data/llama3-8b_top5/marco_LLM_rerank/dpo_top_1-5_dev.jsonl'

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
data = read_jsonl(data_file_path)
new_data = []
for exm in data:
     exm['id'] = '1234'
     new_data.append(exm)

save_path = '/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/train_LLM_data/llama3-8b_top5/marco_LLM_rerank/sft_dev.jsonl'
with open(save_path, "w") as f:
        for item in new_data:
            json.dump(item, f)
            f.write("\n")
print('-finish--')