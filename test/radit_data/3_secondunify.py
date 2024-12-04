import json


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


data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/second/squad_v2.jsonl'
data = read_jsonl(data_file)
new_data = []
for example in data:
    passages = example['passage']
    new_passage = []
    for passage in passages:
        text = str(passage)
        new_passage.append(
        {
            'docid':'none',
            'url':'none',
            'title':'none',
            'headings':'none',
            'segment':text,
            'start_char':0,
            'end_char':0,
            'id':'0',
        }
        )
    example['passage'] = new_passage
    new_data.append(example)
save_data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/second/squad_v2.jsonl'
save_to_jsonl(new_data, save_data_file)
print('ok')