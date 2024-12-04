import json
from datasets import load_dataset

data_files = {}

data_files["train"] = '/data/groups/QY_LLM_Other/meisen/radit/dataset/test.jsonl'
extension = '/data/groups/QY_LLM_Other/meisen/radit/dataset/test.jsonl'.split(".")[-1]

data_files["validation"] = '/data/groups/QY_LLM_Other/meisen/radit/dataset/test.jsonl'
extension = '/data/groups/QY_LLM_Other/meisen/radit/dataset/test.jsonl'.split(".")[-1]
if extension == "jsonl" or extension == "jsonlines":
    extension = "json"
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
data = read_jsonl('/data/groups/QY_LLM_Other/meisen/radit/dataset/test.jsonl')
raw_datasets = load_dataset(extension, data_files=data_files)
print('----')