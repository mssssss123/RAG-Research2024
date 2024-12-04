import copy
import os
import random
from torch.utils.data import Dataset
from typing import Dict, Sequence
from dataclasses import dataclass, field

import torch
import transformers
from torch.utils.data import RandomSampler, DataLoader

from utils.train_utils import get_subfolders, random_element, read_jsonl, subfinder
from src.template import (
    IGNORE_INDEX,
    RESPONSE_START_TOKEN_IDS_pre,
    inst_s_list,
    answer_s_list,
    PROMPT_DICT,
    dataset_type_dict,
    user_tokens,
    assistant_tokens,
    pythia_user_tokens,
    pythia_assistant_tokens, RESPONSE_START_TOKEN_IDS,
    minicpm_multi_choice,
    QA_template,
    minicpm_multi_COT_template,
    COT_question,
    minicpm_QA_COT_template,
    augment_template,
    llama_multi_choice, llama_multi_COT_template, llama_QA_COT_templeta,
    llama_dataset_type_dict,
)

class SFTDataset(Dataset):
    def __init__(self, tokenizer, data_args):
        super(SFTDataset, self).__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_data = self.load_data(self.data_args.train_file)
  
    def load_data(self, train_data_file):
        all_data = []
        dataset_list = get_subfolders(train_data_file)
        for dataset in dataset_list:
            dataset_file = os.path.join(train_data_file,dataset,dataset+'.jsonl')
            data = read_jsonl(dataset_file)
            all_data.extend(data)
        return all_data
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        example = self.train_data[index]
        data_dict = encode_with_prompt_completion_format(example, self.tokenizer, self.data_args.max_seq_length, self.data_args.max_psg_length)
        return data_dict
    
def _tokenize_fn(text: str, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int) -> Dict:
    """Tokenize a list of strings."""
    input_ids = labels = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_seq_length,
            truncation=True,
    ).input_ids

    for idx, ii in enumerate(input_ids):
        if ii[-1] != tokenizer.eos_token_id:
            input_ids[idx][-1] = tokenizer.eos_token_id
            labels[idx][-1] = tokenizer.eos_token_id
            
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()
    # print(input_ids_lens)

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def truncated_passage(passage, tokenizer, truncate_size):
    encoded_passage = tokenizer.encode(passage, add_special_tokens=False)
    truncated_encoded_passage = encoded_passage[:truncate_size]
    decoded_passage = tokenizer.decode(truncated_encoded_passage)
    return decoded_passage

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, max_psg_length):
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
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        cot = example['cot']
        template = template.format(retrieved_passage=passage_text, instructions=query)
        answer = str(cot) + ' ' + str(answer)
    elif task_type == 'Summarization':
        context = example['passage']
        context = truncated_passage(context, tokenizer, max_psg_length)
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(context=context)
    elif task_type == 'Reading-Comprehension':
        query = example['question']
        context = example['passage']
        context = truncated_passage(context, tokenizer, max_psg_length)
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(context=context, question=query)
    elif task_type == 'quarel':
        query = example['question']
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(question=query)
    elif task_type == 'Open-domain-QA':
        passages = example['passage'][:3]
        passage_text = []
        for passage in passages:
            text = passage['text']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        query = example['question']
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(retrieved_passage=passage_text, question=query)

    
    source_text = template
    # tokenizer add_bos=True but add_eos=False
    target_text = str(answer) + tokenizer.eos_token
    examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
    sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

    input_ids = examples_tokenized["input_ids"].flatten()
    source_len = sources_tokenized["input_ids_lens"]
    labels = copy.deepcopy(input_ids)
    labels[ :source_len] = IGNORE_INDEX
    
   
    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten()
    }

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


        return batch

def make_supervised_data_module(tokenizer,
                                data_args):
    """Make dataset and collator for supervised fine-tuning."""
    
    if data_args.dataset_name == 'sft':
        dataset_cls = SFTDataset


    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        data_args=data_args
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
    dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=data_args.per_device_train_batch_size,
        num_workers=data_args.preprocessing_num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collator,
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        data_loader=dataloader,
        )

def preprocess_function_nq(example, tokenizer, max_seq_length, max_psg_length):
    # 目前仅适配nq
    template = PROMPT_DICT['Open-domain-QA']
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)
    passages = example['passage'][:3]
    passage_text = []
    for passage in passages:
        text = passage['text']
        passage_text.append(text)
    passage_text = '\n'.join(passage_text)
    passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
    query = example['input']
    output_text = example['output']
    if isinstance(output_text, list):
        # 目前仅适配nq
        output_text = output_text[0]['answer']
    input_text = template.format(retrieved_passage=passage_text, question=query)
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        if role == "user":
            input_ids += user_tokens + content_ids
            label_ids += [IGNORE_INDEX] * len(user_tokens) + [
                IGNORE_INDEX
            ] * len(content_ids)
        else:
            input_ids += assistant_tokens + content_ids
            label_ids += (
                [IGNORE_INDEX] * len(assistant_tokens)
                + content_ids
            )
      
    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
        )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }



def preprocess_function_radit(example, tokenizer, max_seq_length, max_psg_length, model_name):
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
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        cot = example['cot']
        template = template.format(retrieved_passage=passage_text, instructions=query)
        answer = str(cot) + ' ' + str(answer)
    elif task_type == 'Summarization':
        context = example['passage'][0]['text']
        context = truncated_passage(context, tokenizer, max_psg_length)
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(context=context)
    elif task_type == 'Reading-Comprehension':
        query = example['question']
        context = example['passage'][0]['text']
        context = truncated_passage(context, tokenizer, max_psg_length)
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(context=context, question=query)
    elif task_type == 'quarel':
        query = example['question']
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(question=query)
    elif task_type == 'Open-domain-QA':
        passages = example['passage'][:3]
        passage_text = []
        for passage in passages:
            text = passage['text']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        query = example['question']
        answer = example['answer']
        if isinstance(answer, list):
            answer = answer[0]
        template = template.format(retrieved_passage=passage_text, question=query)

    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        if model_name == 'minicpm':
            if role == "user":
                input_ids += user_tokens + content_ids
                label_ids += [IGNORE_INDEX] * len(user_tokens) + [
                    IGNORE_INDEX
                ] * len(content_ids)
            else:
                input_ids += assistant_tokens + content_ids
                label_ids += (
                    [IGNORE_INDEX] * len(assistant_tokens)
                    + content_ids
                )
        elif model_name == 'pythia':
            if role == "user":
                input_ids += pythia_user_tokens + content_ids
                label_ids += [IGNORE_INDEX] * len(pythia_user_tokens) + [
                    IGNORE_INDEX
                ] * len(content_ids)
            else:
                input_ids += pythia_assistant_tokens + content_ids
                label_ids += (
                    [IGNORE_INDEX] * len(pythia_assistant_tokens)
                    + content_ids
                )
      
    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
        )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }

def get_target(example):
    if 'target' in example:
        target = example['target']
        return target 
    elif 'answers' in example:
        return random.choice(example['answers']) 
    else:
        return None

def preprocess_function_radit_minicpm_v3(example, tokenizer, max_seq_length, max_psg_length, num_passage):  
    data_type = example['data_type']
    task_type = dataset_type_dict[data_type]
    if data_type in ["commonsense_qa","math_qa"]:
        query = minicpm_multi_choice.format(example['question'])
    elif data_type in ["coqa","web_questions","wiki_qa","yahoo_answers_qa","marcoqa"]:
        query = QA_template.format(example['question'])
    elif data_type in ["aqua_rat","ecqa"]:
        query = minicpm_multi_COT_template + ' ' + COT_question.format(example['question'])      
    elif example['data_type'] in ["gsm8k","strategyqa"]:
        query = minicpm_QA_COT_template + ' ' + COT_question.format(example['question'])  

    if len(example['rerank_passage'])>= num_passage:
        passages = example['rerank_passage'][:num_passage]
    else:
        passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  
    passage_text = []
    for passage in passages:
        text = passage['segment']
        passage_text.append(text)
    passage_text = '\n'.join(passage_text)
    passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)

    template = augment_template.format(passage_text, query)

    answer = example['answer']
    if task_type == 'Chain-of-thought-Reasoning':
        cot = example['cot']
        # answer = str(answer) + '. Reason: ' + str(cot)
        answer = str(answer) + '[<COT]' + ' ' + str(cot) + '[COT>]'
    
    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
   
        if role == "user":
            input_ids += user_tokens + content_ids
            label_ids += [IGNORE_INDEX] * len(user_tokens) + [
                IGNORE_INDEX
            ] * len(content_ids)
        else:
            input_ids += assistant_tokens + content_ids
            label_ids += (
                [IGNORE_INDEX] * len(assistant_tokens)
                + content_ids
            )
      
    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
        )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }

def preprocess_function_refinement_llama(example, tokenizer, max_seq_length, max_psg_length, is_instruct, num_passage):  
    query = example['query']
    passage_text = example['passages'][0]['passage_text']
    label = example['label']
    passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
    system_prompt = """Given the following question and context,
                        return YES if the context is relevant to the question and NO if it isn't.

                        > Question: {question}
                        > Context:
                        >>>
                        {context}
                        >>>
                        > Relevant (YES / NO):"""
    template = system_prompt.format(question = query, context = passage_text)

    input_text = template
    output_text = label
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    
    input_ids = tokenizer.apply_chat_template(messages)
    if is_instruct:
        trigger = RESPONSE_START_TOKEN_IDS
    else:
        trigger = RESPONSE_START_TOKEN_IDS_pre
    response_start_pos = subfinder(input_ids, trigger)
    assert response_start_pos != -1
    response_start_pos += len(trigger)
    label_ids = [-100] * response_start_pos + input_ids[response_start_pos:]

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
    )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }
def preprocess_function_radit_llama_v3(example, tokenizer, max_seq_length, max_psg_length, is_instruct, num_passage):  
    data_type = example['data_type']
    task_type = llama_dataset_type_dict[data_type]
    if data_type in ['commonsense_qa', 'math_qa',"aqua_rat","ecqa"]:
        query = llama_multi_choice.format(example['question'])
    elif data_type in ["coqa","web_questions","wiki_qa","yahoo_answers_qa","marcoqa"]:
        query = QA_template.format(example['question'])
    # elif data_type in ["aqua_rat","ecqa"]:
    #     query = llama_multi_COT_template + ' ' + COT_question.format(example['question'])      
    elif example['data_type'] in ["gsm8k","strategyqa"]:
        query = llama_QA_COT_templeta + ' ' + COT_question.format(example['question'])  

    if len(example['rerank_passage'])>= num_passage:
        passages = example['rerank_passage'][:num_passage]
    else:
        passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  
    passage_text = []
    for passage in passages:
        text = passage['segment']
        passage_text.append(text)
    passage_text = '\n'.join(passage_text)
    passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)

    template = augment_template.format(passage_text, query)

    answer = example['answer']
    if task_type == 'Chain-of-thought-Reasoning':
        cot = example['cot']
        # answer = str(answer) + '. Reason: ' + str(cot)
        answer = str(answer) + '[<COT]' + ' ' + str(cot) + '[COT>]'
    
    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = tokenizer.apply_chat_template(messages)
    if is_instruct:
        trigger = RESPONSE_START_TOKEN_IDS
    else:
        trigger = RESPONSE_START_TOKEN_IDS_pre
    response_start_pos = subfinder(input_ids, trigger)
    assert response_start_pos != -1
    response_start_pos += len(trigger)
    label_ids = [-100] * response_start_pos + input_ids[response_start_pos:]

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
    )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }
def preprocess_function_radit_minicpm_dpo(example, tokenizer, max_seq_length, max_psg_length, num_passage):  
    data_type = example['data_type']
    task_type = dataset_type_dict[data_type]
    template = PROMPT_DICT[task_type]
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)

   
    if task_type == 'Summarization':

        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]

        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text)

    elif task_type == 'Reading-Comprehension' or task_type == 'Open-domain-QA' or task_type == 'Chain-of-thought-Reasoning':
        query = example['question']

        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]

        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text, question=query)
        if task_type == 'Chain-of-thought-Reasoning':
            cot = example['cot']
            answer = str(cot) + ' ' + str(answer)

    elif task_type == 'quarel':
        query = example['question']
        answer = example['answer']
        template = template.format(question=query)

    dpo_chosen_text = example['chosen']['text']
    input_text = template
    output_text = dpo_chosen_text
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
      
        if role == "user":
            input_ids += user_tokens + content_ids
            label_ids += [IGNORE_INDEX] * len(user_tokens) + [
                IGNORE_INDEX
            ] * len(content_ids)
        else:
            input_ids += assistant_tokens + content_ids
            label_ids += (
                [IGNORE_INDEX] * len(assistant_tokens)
                + content_ids
            )
    
      
    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
        )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }
 
def preprocess_function_radit_minicpm_v2(example, tokenizer, max_seq_length, max_psg_length, num_passage):  
    data_type = example['data_type']
    task_type = dataset_type_dict[data_type]
    template = PROMPT_DICT[task_type]
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)

   
    if task_type == 'Summarization':

        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]

        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text)

    elif task_type == 'Reading-Comprehension' or task_type == 'Open-domain-QA' or task_type == 'Chain-of-thought-Reasoning':
        query = example['question']

        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]

        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text, question=query)
        if task_type == 'Chain-of-thought-Reasoning':
            cot = example['cot']
            answer = str(cot) + ' ' + str(answer)

    elif task_type == 'quarel':
        query = example['question']
        answer = example['answer']
        template = template.format(question=query)

    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
      
        if role == "user":
            input_ids += user_tokens + content_ids
            label_ids += [IGNORE_INDEX] * len(user_tokens) + [
                IGNORE_INDEX
            ] * len(content_ids)
        else:
            input_ids += assistant_tokens + content_ids
            label_ids += (
                [IGNORE_INDEX] * len(assistant_tokens)
                + content_ids
            )
    
      
    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
        )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }
def preprocess_function_radit_v2_chatqa(example, tokenizer, max_seq_length, max_psg_length, is_instruct, num_passage):  
    data_type = example['data_type']
    task_type = dataset_type_dict[data_type]
    template = PROMPT_DICT[task_type]
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)

   
    if task_type == 'Summarization':
        passages = example['passage'][:num_passage]
        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text)

    elif task_type == 'Reading-Comprehension' or task_type == 'Open-domain-QA' or task_type == 'Chain-of-thought-Reasoning':
        query = example['question']
        passages = example['passage'][:num_passage]
        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text, question=query)
        if task_type == 'Chain-of-thought-Reasoning':
            cot = example['cot']
            answer = str(cot) + ' ' + str(answer)

    elif task_type == 'quarel':
        query = example['question']
        answer = example['answer']
        template = template.format(question=query)

    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    system_ids = tokenizer.encode(system, add_special_tokens=False)
    input_ids += system_ids 
    label_ids += [IGNORE_INDEX] * len(system_ids) 
 
    the_user_tokens = tokenizer.encode('User:', add_special_tokens=False)
    the_assistant_tokens = tokenizer.encode('Assistant:', add_special_tokens=False)
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        
        if role == "user":
            input_ids += the_user_tokens + content_ids
            label_ids += [IGNORE_INDEX] * len(the_user_tokens) + [
                IGNORE_INDEX
            ] * len(content_ids)
        else:
            input_ids += the_assistant_tokens + content_ids
            label_ids += (
                [IGNORE_INDEX] * len(the_assistant_tokens)
                + content_ids
            )
  
    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
        )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }

def preprocess_function_radit_llama_dpo(example, tokenizer, max_seq_length, max_psg_length, is_instruct, num_passage):  
    data_type = example['data_type']
    task_type = dataset_type_dict[data_type]
    template = PROMPT_DICT[task_type]
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)

   
    if task_type == 'Summarization':

        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]
        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text)

    elif task_type == 'Reading-Comprehension' or task_type == 'Open-domain-QA' or task_type == 'Chain-of-thought-Reasoning':
        query = example['question']
        
        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]
        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text, question=query)
        if task_type == 'Chain-of-thought-Reasoning':
            cot = example['cot']
            answer = str(cot) + ' ' + str(answer)

    elif task_type == 'quarel':
        query = example['question']
        answer = example['answer']
        template = template.format(question=query)
    
    dpo_chosen_text = example['chosen']['text']
    input_text = template
    output_text = dpo_chosen_text
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    # a = tokenizer.encode('<|im_start|>assistant')
    # b = tokenizer.decode(RESPONSE_START_TOKEN_IDS_pre)
    # c_message = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer.apply_chat_template(messages)
    if is_instruct:
        trigger = RESPONSE_START_TOKEN_IDS
    else:
        trigger = RESPONSE_START_TOKEN_IDS_pre
    response_start_pos = subfinder(input_ids, trigger)
    assert response_start_pos != -1
    response_start_pos += len(trigger)
    label_ids = [-100] * response_start_pos + input_ids[response_start_pos:]

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
    )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }

def preprocess_function_radit_llama_v2(example, tokenizer, max_seq_length, max_psg_length, is_instruct, num_passage):  
    data_type = example['data_type']
    task_type = dataset_type_dict[data_type]
    template = PROMPT_DICT[task_type]
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)

   
    if task_type == 'Summarization':

        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]
        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text)

    elif task_type == 'Reading-Comprehension' or task_type == 'Open-domain-QA' or task_type == 'Chain-of-thought-Reasoning':
        query = example['question']
        
        if len(example['rerank_passage'])>= num_passage:
            passages = example['rerank_passage'][:num_passage]
        else:
            passages = example['rerank_passage'] + example['passage'][:num_passage-len(example['rerank_passage'])]  

        # passages = example['passage'][:num_passage]
        passage_text = []
        for passage in passages:
            text = passage['segment']
            passage_text.append(text)
        passage_text = '\n'.join(passage_text)
        passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
        answer = example['answer']
        template = template.format(context=passage_text, question=query)
        if task_type == 'Chain-of-thought-Reasoning':
            cot = example['cot']
            answer = str(cot) + ' ' + str(answer)

    elif task_type == 'quarel':
        query = example['question']
        answer = example['answer']
        template = template.format(question=query)

    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    # a = tokenizer.encode('<|im_start|>assistant')
    # b = tokenizer.decode(RESPONSE_START_TOKEN_IDS_pre)
    # c_message = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer.apply_chat_template(messages)
    if is_instruct:
        trigger = RESPONSE_START_TOKEN_IDS
    else:
        trigger = RESPONSE_START_TOKEN_IDS_pre
    response_start_pos = subfinder(input_ids, trigger)
    assert response_start_pos != -1
    response_start_pos += len(trigger)
    label_ids = [-100] * response_start_pos + input_ids[response_start_pos:]

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
    )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }

def preprocess_function_fid(example, tokenizer, max_seq_length, max_psg_length, model_name, num_passage):
    question = "question:" + " " + example['question'] + " " + "answer:"
    target = get_target(example)
    f = "title:" + " {} " + "context:" + " {}"
    if 'score' in example['ctxs'][0]:
        example['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)
    contexts = example['ctxs'][:num_passage]
    passages = [f.format(c['title'], c['text']) for c in contexts]
    passages = " ".join(passages)   
    passages = truncated_passage(passages, tokenizer, max_psg_length)
    input_text = passages + " " + question
    output_text = target
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        if model_name == 'minicpm':
            if role == "user":
                input_ids += user_tokens + content_ids
                label_ids += [IGNORE_INDEX] * len(user_tokens) + [
                    IGNORE_INDEX
                ] * len(content_ids)
            else:
                input_ids += assistant_tokens + content_ids
                label_ids += (
                        [IGNORE_INDEX] * len(assistant_tokens)
                        + content_ids
                )
        elif model_name == 'pythia':
            if role == "user":
                input_ids += pythia_user_tokens + content_ids
                label_ids += [IGNORE_INDEX] * len(pythia_user_tokens) + [
                    IGNORE_INDEX
                ] * len(content_ids)
            else:
                input_ids += pythia_assistant_tokens + content_ids
                label_ids += (
                        [IGNORE_INDEX] * len(pythia_assistant_tokens)
                        + content_ids
                )

    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
    )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }

def preprocess_function(example, tokenizer, max_seq_length, max_psg_length, model_name):
    # 适配marco qa
    template = PROMPT_DICT['Open-domain-QA']
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)
    # 目前写死了，top3
    top_k_passage = example['passage'][:3]
    passage_text = []
    for passage in top_k_passage:
        text = passage['segment']
        passage_text.append(text)
    passage_text = '\n'.join(passage_text)
    passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
    query = example['query']
    answer = example['answers']
    if isinstance(answer, list):
        answer = answer[0]
    template = template.format(retrieved_passage=passage_text, question=query)

    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = [tokenizer.bos_token_id]
    label_ids = [IGNORE_INDEX]
    for message in messages:
        role = message["role"]
        content = message["content"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        if model_name == 'minicpm':
            if role == "user":
                input_ids += user_tokens + content_ids
                label_ids += [IGNORE_INDEX] * len(user_tokens) + [
                    IGNORE_INDEX
                ] * len(content_ids)
            else:
                input_ids += assistant_tokens + content_ids
                label_ids += (
                        [IGNORE_INDEX] * len(assistant_tokens)
                        + content_ids
                )
        elif model_name == 'pythia':
            if role == "user":
                input_ids += pythia_user_tokens + content_ids
                label_ids += [IGNORE_INDEX] * len(pythia_user_tokens) + [
                    IGNORE_INDEX
                ] * len(content_ids)
            else:
                input_ids += pythia_assistant_tokens + content_ids
                label_ids += (
                        [IGNORE_INDEX] * len(pythia_assistant_tokens)
                        + content_ids
                )

    input_ids.append(tokenizer.eos_token_id)
    label_ids.append(tokenizer.eos_token_id)

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
    )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }

def preprocess_function_llama(example, tokenizer, max_seq_length, max_psg_length, model_name):
    # 适配marco qa
    template = PROMPT_DICT['Open-domain-QA']
    inst = random_element(inst_s_list)
    answ = random_element(answer_s_list)
    template = template.replace("<inst s>", inst)
    template = template.replace("<answer s>", answ)
    # 目前写死了，top3
    top_k_passage = example['passage'][:3]
    passage_text = []
    for passage in top_k_passage:
        text = passage['segment']
        passage_text.append(text)
    passage_text = '\n'.join(passage_text)
    passage_text = truncated_passage(passage_text, tokenizer, max_psg_length)
    query = example['query']
    answer = example['answers']
    if isinstance(answer, list):
        answer = answer[0]
    template = template.format(retrieved_passage=passage_text, question=query)

    input_text = template
    output_text = answer
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    input_ids = tokenizer.apply_chat_template(messages)
    response_start_pos = subfinder(input_ids, RESPONSE_START_TOKEN_IDS)
    assert response_start_pos != -1
    response_start_pos += len(RESPONSE_START_TOKEN_IDS)
    label_ids = [-100] * response_start_pos + input_ids[response_start_pos:]

    input_ids = input_ids[: max_seq_length]
    label_ids = label_ids[: max_seq_length]
    attention_mask = [1] * len(input_ids)
    input_ids += [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
    )
    label_ids += [IGNORE_INDEX] * (max_seq_length - len(label_ids))
    attention_mask += [0] * (max_seq_length - len(attention_mask))
    input_ids = torch.LongTensor(input_ids)
    label_ids = torch.LongTensor(label_ids)
    attention_mask = torch.LongTensor(attention_mask)
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }