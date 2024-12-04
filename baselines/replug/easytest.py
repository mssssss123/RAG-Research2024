

import argparse
import csv

import jsonlines
import numpy as np
import torch
from tqdm import tqdm

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import softmax
from collections import defaultdict
import operator

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data
def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["output"]:
        if isinstance(item, str):
            ground_truths.add(item)
        elif "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())

    return ground_truths

def read_csv_to_list(filepath, delimiter=' '):
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        data_list = [row for row in reader]  # 将每一行数据作为列表添加到data_list中
    return data_list
def chunk_list(original_list, chunk_size):
    # 使用列表推导式创建新的分块列表
    return [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='/data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16')
    parser.add_argument('--input_file', type=str, default='/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl')
    parser.add_argument('--passage_score_file', type=str, default='/data/groups/QY_LLM_Other/meisen/iclr2024/replug/process_trec_file/trex_dev.trec')
    parser.add_argument('--case_num', type=int, default=-1)
    parser.add_argument('--pred_len', type=int, default=32)
    
    return parser.parse_args()


def process_single_example(example, psg_score_dict, tokenizer, model,max_new_tokens):
    # 数据处理
    query = example['input']
    answer = list(get_gold_answers(example))[0]
    passages = example['rerank_passage'][:5]
    prompt_batch = []
    score_batch = []
    prompt = ""
    for doc in passages:
        prompt_cur = prompt
        docid = doc['docid']
        docscore = psg_score_dict[docid]
        text = doc['segment']
        prompt_cur += f"Knowledge: {text}" + "\n"
        prompt_cur += "Question: " + query  + "\n"
        prompt_cur += "Answer:"
        prompt_batch.append(prompt_cur)
        score_batch.append(docscore)
    score_batch = softmax(np.array(score_batch)).tolist()

    # ppl计算
    all_outputs = []
    all_probs = []
    for iindex, prompt in enumerate(prompt_batch):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_length = input_ids.shape[-1]
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                return_dict_in_generate=True, 
                output_scores=True, 
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                output_logits=True,
                do_sample=False,
            )
        token_ids = output.sequences[0]
       
        log_probs = []
        for logits in output.scores:  
            log_probs.append(torch.nn.functional.log_softmax(logits, dim=-1))   
        log_probs = torch.cat(log_probs, dim=0) 
        
        generated_token_ids = token_ids[input_length:]  # 只保留生成的 token
        token_log_probs = []
       
       
        for j, token_id in enumerate(generated_token_ids):
            token_log_prob = log_probs[j, token_id].item()  
            token_log_probs.append(token_log_prob)
        # ppl的倒数
        perplexity = np.exp(np.mean(token_log_probs))
        output_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        all_outputs.append(output_text)
        all_probs.append(perplexity*score_batch[iindex])
    ans2prob_list = defaultdict(list)
    for ans, prob in zip(all_outputs, all_probs):
        ans2prob_list[ans].append(prob)
    ans2prob = {k: sum(v) for k, v in ans2prob_list.items()}
    # bp()
    final_ans = max(ans2prob.items(), key=operator.itemgetter(1))[0]
    print('---')

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left",
                                                    truncation_side="right", )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='cuda', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]

    trec_list = read_csv_to_list(args.passage_score_file)
    sub_trec=chunk_list(trec_list,100)

    for index, example in tqdm(enumerate(input_data)):
        cur_example_trec = sub_trec[index]
        cur_psg_score_dict = {}
        for tt in cur_example_trec:
            doc_id = str(tt[2])
            score = float(tt[4])
            cur_psg_score_dict[doc_id] = score

        process_single_example(example,cur_psg_score_dict,tokenizer,model,args.pred_len)
        print('----')







if __name__ == "__main__":
    main()