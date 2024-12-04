import argparse
import json
import os
import pdb
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

user_tokens='<用户>',
assistant_tokens='<AI>'
class LLMReranker:

    def __init__(
            self,
            args = None,
            model_name_or_path: str = 'BAAI/bge-reranker-v2-minicpm-layerwise',
            topn: int = 10,
            needn:int =5):
        
        model = LLM(model= model_name_or_path, tensor_parallel_size= 1, trust_remote_code=True,)
                    #dtype='bfloat16',)
        params_dict = {
                "n": 1,
                "best_of": 1,
                "presence_penalty": 1.0,
                "frequency_penalty": 0.0,
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": 1,
                "use_beam_search": False,
                "length_penalty": 1,
                "early_stopping": False,
                "stop": None,
                "stop_token_ids": None,
                "ignore_eos": False,
                "max_tokens": 32,
                "logprobs": None,
                "prompt_logprobs": None,
                "skip_special_tokens": True,
            }

        # Create a sampling params object.
        sampling_params = SamplingParams(**params_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                                    padding_side="left",truncation_side="right",is_pretrained_tokenizer=True )
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer =tokenizer

        self.model = model
        self.topn = topn
        self.sampling_params =sampling_params
        self.needn = needn
        self.args = args
        self.system_prompt = """Given the following question and context,
return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):"""

#return YES if the context is relevant to the question and the context can answer the question, and return NO if it isn't or can't answer.
        self.rerank_prompt = "Please ignore aspects such as length, complexity or writing style of passage B  and concentrate on determining whether the passage B contains the necessary information needed to be able to answer the query A."
        self.first_prompt = "Output: ['Yes' or 'No']"

        

    def get_inputs(self, pairs, prompt=None, max_length=1024):
        """Build input tokens with query and chunks."""
        if prompt is None:
            # prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            prompt = self.system_prompt
        sep = '\n'
        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            
            if self.args.task == "t-rex":
                query = "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity. {}\nAnswer:".format(
                    query)
            # query_inputs = self.tokenizer(f'A: {query}',
            #                               return_tensors=None,
            #                               add_special_tokens=False,
            #                               max_length=max_length * 3 // 4,
            #                               truncation=True)
            # passage_inputs = self.tokenizer(f'B: {passage}',
            #                                 return_tensors=None,
            #                                 add_special_tokens=False,
            #                                 max_length=max_length,
            #                                 truncation=True)
            # item = self.tokenizer.prepare_for_model(
            #     [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
            #     sep_inputs + passage_inputs['input_ids'],
            #     truncation='only_second',
            #     max_length=max_length,
            #     padding=False,
            #     return_attention_mask=False,
            #     return_token_type_ids=False,
            #     add_special_tokens=False)

            # item_input_ids = item['input_ids'] + sep_inputs + prompt_inputs
            # new_prompt = self.tokenizer.decode(item_input_ids,skip_special_tokens=True)

            passage_inputs = self.tokenizer(passage,
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length,
                                            truncation=True)['input_ids']
            new_passage = self.tokenizer.decode(passage_inputs,skip_special_tokens=True)
            new_prompt = self.system_prompt.format(question = query, context = new_passage)

            if self.args.llama_style:
                messages = [
                    {"role": "user", "content": new_prompt},
                ]
                item_input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:

                item_input_ids = "<用户>{}<AI>".format(new_prompt)

            inputs.append(item_input_ids)
        return inputs

    def split_list(self,input_list, n):
        return [input_list[i:i + n] for i in range(0, len(input_list), n)]
    
    def judege_output(self,pred):
        return 'YES' in pred or 'yes' in pred or 'Yes' in pred 


    def inference(self, chunks, query_list):
        """Rerank input chunks, return descending indexes, indexes[0] is the
        nearest chunk."""
        pairs = []
        for idx, query in enumerate(query_list):
            for chunk in chunks[idx]:
                pairs.append([query, chunk])

        split_chunks = self.split_list(pairs,self.needn)
        count_of_ones = 0
        judge_preds = []
        preds = []

        for sub_chunk in split_chunks:
            inputs = self.get_inputs(sub_chunk)
            outputs = self.model.generate(inputs, self.sampling_params)
            for pred in outputs:
                pred = pred.outputs[0].text.lstrip()
                judge_preds.append(self.judege_output(pred))
                if self.judege_output(pred) == True:
                    count_of_ones +=1

                preds.append(pred)

            if count_of_ones >= self.needn:
                break

        return judge_preds



def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def split_list(lst, n):
    array = np.array(lst)
    return np.array_split(array, n)


def get_batch_input(data, batch_size):
    split_data =[]
    split_array = split_list(data, np.ceil(len(data) / batch_size))
    for arr in split_array:
        split_data.append(arr.tolist())

    return split_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file_path', type=str,
                        default="/home/lixz23/ragsft/data/marco_v2.1/bge_large_retriever_128_256_top100/hotpotqa_dev_psg.jsonl")
    parser.add_argument('--model_name_or_path', type=str,
                        default="/home/lixz23/ragsft/DPO/icrl2024_checkpoint/LLM_rerank/top_20_random/merge-500")
    parser.add_argument('--output_path', type=str,
                        default="/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/top_50_trained_marco/nq_list")
    parser.add_argument('--file_name', type=str,
                        default="nq_dev_psg.jsonl")
    parser.add_argument('--top_n', type=int,
                        default=100)
    
    parser.add_argument('--need_n', type=int,
                        default=5)
    
    parser.add_argument('--cut_num', type=int,default=1)
    parser.add_argument('--number', type=int,default=0)
    parser.add_argument('--batch_size', type=int,default=1)
    parser.add_argument('--task', type=str,default='nq')

    parser.add_argument('--llama_style', action='store_true',default=True)
    
    args = parser.parse_args()
    print("模型的所有参数:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    reranker = LLMReranker(args = args, model_name_or_path=args.model_name_or_path, topn=args.top_n,needn=args.need_n)

    raw_data = read_jsonl(args.dataset_file_path)
    raw_data_list = split_list(raw_data, args.cut_num)
    data = raw_data_list[args.number].tolist()

    data = get_batch_input(data, args.batch_size)


    for item in tqdm(data, desc="Processing examples"):
        question_list = []
        passages_list = []
        for example in item:
            if 'input' in example:
                question = example['input']
            elif 'query' in example:
                question = example['query']
            elif 'question' in example:
                question = example['question']
            question_list.append(question)
            passages = []
            for psg in example['passage']:
                segment = psg['segment']
                passages.append(segment)

            passages = passages[:args.top_n]
            passages_list.append(passages)
            
        judge_preds = reranker.inference(chunks=passages_list, query_list=question_list)
        example['judge_preds'] = judge_preds
  
        for example in item:
            example['rerank_passage'] = []
            if any(judge_preds):
                for judge,psg in zip(judge_preds,example['passage']):
                    if judge:
                        example['rerank_passage'].append(psg)
            else:
                example['rerank_passage'] = example['passage']
      
    data = sum(data, [])
    output_path = os.path.join(args.output_path, args.file_name)
    with open(output_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
    print('finish')
    

    
    

if __name__ == "__main__":
    main()