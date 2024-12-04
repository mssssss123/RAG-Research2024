
import json
import os
import random

import torch
random.seed(7)
inst_s_list = ['Q:', 'Question:', '', 'Input:']
answer_s_list = ['A:', 'Answer:', 'Response:']
PROMPT_DICT = {
    "Open-domain-QA": (
        "Background:\n{retrieved_passage}\n\n<inst s>\n{question}\n\n<answer s>\n"
    ),
    "Reading-Comprehension": (
        "Background:\n{context}\n\n<inst s>\n{question}\n\n<answer s>\n"
    ),
    # 这个数据集很特殊，没有passage
    "quarel": (
        "<inst s>\n{question}\n\n<answer s>\n"
    ),
    "Summarization": (
        "Background:\n{context}\n\nSummarize this article:\n<answer s>\n"
    ),
    # 我对这个任务有点问题了，cot已经包含答案了
    "Chain-of-thought-Reasoning": (
        "Background:\n{retrieved_passage}\n\n<inst s>\n{instructions}\n\n<answer s>\n"
    ),
}
dataset_type_dict = {
    "aqua_rat" : "Chain-of-thought-Reasoning",
    "cnn_dailymail" : "Summarization",
    "commonsense_qa" : "Open-domain-QA",
    "coqa" : "Reading-Comprehension",
    "drop" : "Reading-Comprehension",
    "ecqa" : "Chain-of-thought-Reasoning",
    "gsm8k" : "Chain-of-thought-Reasoning",
    "math_qa" : "Open-domain-QA",
    "newsqa" : "Reading-Comprehension",
    "quail" : "Reading-Comprehension",
    "quarel" : "quarel",
    "squad_v2" : "Reading-Comprehension",
    "strategyqa" : "Chain-of-thought-Reasoning",
    "web_questions" : "Open-domain-QA",
    "wiki_qa" : "Open-domain-QA",
    "yahoo_answers_qa" : "Open-domain-QA",
}

def random_element(array):
    return random.choice(array)

def get_files_in_path(path):
    files_array = []
    for root, dirs, files in os.walk(path):
        for file in files:
            files_array.append(os.path.join(root, file))
    return files_array

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
def get_subfolders(path):

    all_items = os.listdir(path)

    sub_folders = [item for item in all_items if os.path.isdir(os.path.join(path, item))]

    return sub_folders

def main():

    a = read_jsonl('/data/groups/QY_LLM_Other/lixinze/rag_train_with_psg_all/debug.jsonl')


    qa = 0
    sum = 0
    read = 0
    cot = 0

    # 以下是读取代码
    all_data_path = '/data/groups/QY_LLM_Other/lixinze/rag_train_with_psg'
    all_data = []
    dataset_list = get_subfolders(all_data_path)
    for dataset in dataset_list:
        dataset_file = os.path.join(all_data_path,dataset,dataset+'.jsonl')
        data = read_jsonl(dataset_file)
        task_type = dataset_type_dict[dataset]
        if task_type == 'Open-domain-QA':
            qa += len(data)
        elif task_type == 'Reading-Comprehension':
            read += len(data)
        elif task_type == 'Summarization':
            sum += len(data)
        elif task_type == 'Chain-of-thought-Reasoning':
            cot += len(data)
        elif task_type == 'quarel':
            read += len(data)
        all_data.append(data)
        print('----')




    # 以下是处理代码
    file = '/data/groups/QY_LLM_Other/lixinze/rag_train_with_psg/aqua_rat/aqua_rat.jsonl'
    data = read_jsonl(file)
    for example in data:
        data_type = example['data_type']
        task_type = dataset_type_dict[data_type]
        template = PROMPT_DICT[task_type]
        inst = random_element(inst_s_list)
        answ = random_element(answer_s_list)
        template = template.replace("<inst s>", inst)
        template = template.replace("<answer s>", answ)
        if task_type == 'Chain-of-thought-Reasoning':
            query = example['question']
            passages = example['passage'][:3]
            passage_text = []
            for passage in passages:
                text = passage['text']
                passage_text.append(text)
            passage_text = '\n'.join(passage_text)
            answer = example['answer']
            if isinstance(answer, list):
                answer = answer[0]
            cot = example['cot']
            template = template.format(retrieved_passage=passage_text, instructions=query)
            answer = cot + ' ' + answer
            print('----')
        elif task_type == 'Summarization':
            context = example['passage']
            answer = example['answer']
            if isinstance(answer, list):
                answer = answer[0]
            template = template.format(context=context)
            print('-----')
        elif task_type == 'Reading-Comprehension':
            query = example['question']
            context = example['passage']
            answer = example['answer']
            if isinstance(answer, list):
                answer = answer[0]
            template = template.format(context=context, question=query)
            print('-----')
        elif task_type == 'quarel':
            query = example['question']
            answer = example['answer']
            if isinstance(answer, list):
                answer = answer[0]
            template = template.format(question=query)
            print('-----')
        elif task_type == 'Open-domain-QA':
            passages = example['passage'][:3]
            passage_text = []
            for passage in passages:
                text = passage['text']
                passage_text.append(text)
            passage_text = '\n'.join(passage_text)
            query = example['question']
            answer = example['answer']
            if isinstance(answer, list):
                answer = answer[0]
            template = template.format(retrieved_passage=passage_text, question=query)
            print('---')








if __name__ == "__main__":
    main()