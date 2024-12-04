

import argparse

import jsonlines
from tqdm import tqdm

import models
import json

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

def compute_perplexity_data(model, data=None, args=None):
    def save_prob(output):
        overall_output["all_logprobs"].append(output["logprobs"])
        overall_output["all_positions"].append(output["positions"])
        overall_output["aggregate_length"] += output["length"]
        overall_output["aggregate_utf8_length"] += output["utf8_length"]

    overall_output = {
        "all_logprobs": [],
        "all_positions": [],
        "aggregate_length": 0,
        "aggregate_utf8_length": 0.
    }
    for i, example in tqdm(enumerate(data)):
        output = model.get_perplexity_data(example)
        if not output:
                continue
        save_prob(output)
        
    if args.data.startswith("wikitext"):
        data = load_dataset("wikitext", args.data, split=f"test[0%:{int(args.data_ratio*100)}%]")
        data = data["text"]
        for i in tqdm(range(0, len(data), 10000)):
            batch = data[i:i+10000]
            doc = "\n\n".join(batch)
            output = model.get_perplexity_data(doc)
            if not output:
                continue
            save_prob(output)
    else:
        reader = lm_dataformat.Reader(args.data)
        embed()  # set ratio
        for i, doc in enumerate(tqdm_lib.tqdm(reader.stream_data())):
            if indices is not None and i not in indices:
                continue
            output = model.get_perplexity_data(doc)
            if not output:
                continue
            save_prob(output)
    return overall_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='/data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16')
    parser.add_argument('--input_file', type=str, default='/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl')
    parser.add_argument('--case_num', type=int, default=-1)
    parser.add_argument('--context_len', type=int, default=256)
    parser.add_argument('--pred_len', type=int, default=256)
    
    return parser.parse_args()







def main():
    args = parse_args()
    model = models.create_model(args)
    model.context_len = args.context_len
    model.max_seq_len = args.context_len + args.pred_len
    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]
    perplexity_data = compute_perplexity_data(model=model, data=input_data, args=args)
    print('----')







if __name__ == "__main__":
    main()