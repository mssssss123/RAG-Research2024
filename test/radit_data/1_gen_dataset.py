import json
import os

other_dataset_dict = {
    "cnn_dailymail" : "Summarization",
    "coqa" : "Reading-Comprehension",
    "drop" : "Reading-Comprehension",
    "newsqa" : "Reading-Comprehension",
    "quail" : "Reading-Comprehension",
    "quarel" : "quarel",
    "squad_v2" : "Reading-Comprehension",
}

qa_dataset_dict = {
    "commonsense_qa" : "Open-domain-QA",
    "math_qa" : "Open-domain-QA",
    "web_questions" : "Open-domain-QA",
    "wiki_qa" : "Open-domain-QA",
    "yahoo_answers_qa" : "Open-domain-QA",
    "aqua_rat" : "Chain-of-thought-Reasoning",
    "ecqa" : "Chain-of-thought-Reasoning",
    "gsm8k" : "Chain-of-thought-Reasoning",
    "strategyqa" : "Chain-of-thought-Reasoning",

}

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

# 按照数据类型把sub_radit_psg.jsonl切分回不同的qa数据集文件
def split_whole_qadataset(data):
    save_root_path = '/data/groups/QY_LLM_Other/meisen/radit/dataset/first'
    new_data = {}
    for example in data:
        data_type = example['data_type']
        if data_type not in new_data:
            new_data[data_type] = []
        new_data[data_type].append(example)
    new_data_keys = new_data.keys()
    new_data_keys = list(new_data_keys)
    for new_data_key in new_data_keys:
        data = new_data[new_data_key]
        save_path = os.path.join(save_root_path,new_data_key+'.jsonl')
        save_to_jsonl(data,save_path)
    print('all finish!')

def main():
    other_data_root_path = '/data/groups/QY_LLM_Other/lixinze/rag_train'
    qa_data_file = '/home/lixz23/ragsft/data/radit/sub_data_need_psg/sub_radit_psg.jsonl'
    data = read_jsonl(qa_data_file)
    split_whole_qadataset(data)
    print('---')








if __name__ == "__main__":
    main()