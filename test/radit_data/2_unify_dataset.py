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

# cot
# data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/first/strategyqa.jsonl'
# data = read_jsonl(data_file)
# new_data = []
# for example in data:
#     question = example['question']
#     answer = str(example['answer'])
#     passage = example['passage']
#     data_type = example['data_type']
#     cot = example['cot']
#     new_data.append(
#         {
#             'question':question,
#             'answer':answer,
#             'passage':passage,
#             'data_type':data_type,
#             'cot':cot,
#         }
#     )
# save_data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/second/strategyqa.jsonl'
# save_to_jsonl(new_data, save_data_file)

# qa
# web_question是 answer list
# data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/first/train_v2.1_marco2.1_psg.jsonl'
# data = read_jsonl(data_file)
# new_data = []
# for example in data:
#     question = example['query']
#     answer = str(example['answers'][0])
#     passage = example['passage']
#     data_type = 'marcoqa'
#     cot = 'none'
#     new_data.append(
#         {
#             'question':question,
#             'answer':answer,
#             'passage':passage,
#             'data_type':data_type,
#             'cot':cot,
#         }
#     )
# save_data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/second/marco_qa.jsonl'
# save_to_jsonl(new_data, save_data_file)


# Summarization

data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/first/cnn_dailymail.jsonl'
data = read_jsonl(data_file)
new_data = []
for example in data:
    question = 'none'
    answer = str(example['answer'])
    passage = [example['passage']]
    data_type = example['data_type']
    cot = 'none'
    new_data.append(
        {
            'question':question,
            'answer':answer,
            'passage':passage,
            'data_type':data_type,
            'cot':cot,
        }
    )
save_data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/second/cnn_dailymail.jsonl'
save_to_jsonl(new_data, save_data_file)

# read

# data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/first/squad_v2.jsonl'
# data = read_jsonl(data_file)
# new_data = []
# for example in data:
#     question = example['question']
#     answer = str(example['answer'][0])
#     passage = [example['passage']]
#     data_type = example['data_type']
#     cot = 'none'
#     new_data.append(
#         {
#             'question':question,
#             'answer':answer,
#             'passage':passage,
#             'data_type':data_type,
#             'cot':cot,
#         }
#     )
# save_data_file = '/data/groups/QY_LLM_Other/meisen/radit/dataset/second/squad_v2.jsonl'
# save_to_jsonl(new_data, save_data_file)
print('---')