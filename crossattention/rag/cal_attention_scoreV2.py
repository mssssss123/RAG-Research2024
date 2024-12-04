import argparse
import json
import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
def truncated_passage(passage, tokenizer, truncate_size):
    encoded_passage = tokenizer.encode(passage, add_special_tokens=False)
    truncated_encoded_passage = encoded_passage[:truncate_size]
    decoded_passage = tokenizer.decode(truncated_encoded_passage)
    return decoded_passage
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default='/home/lixz23/ragsft/DPO/icrl2024_checkpoint/llama-8b-sft/LLM-rerank/top_1-5/merge-2200')
    parser.add_argument('--input_file', type=str, default='/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl')
    parser.add_argument('--top_n', type=int, default=3,help="number of paragraphs to be considered.")
    parser.add_argument('--case_num', type=int, default=50)
    parser.add_argument('--psg_maxlen', type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left",
                                                    truncation_side="right", )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='cuda', trust_remote_code=True, torch_dtype=torch.bfloat16,)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()


    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]
    
    psg_attention_means1 = []  
    psg_attention_means2 = []  
    psg_attention_means3 = []  
    query_attention_means = []
    for example in tqdm(input_data):
        query = example['input']
        answer = list(get_gold_answers(example))[0]
        if len(example['rerank_passage'])>= args.top_n:
            passages = example['rerank_passage'][:args.top_n]
        passage_text = []
        for passage in passages:
            text = passage['segment']
            text = truncated_passage(text, tokenizer, args.psg_maxlen)
            passage_text.append(text)
        passage1 = 'Background:\n' + passage_text[0]
        passage2 = 'Background:\n' + passage_text[1]
        passage3 = 'Background:\n' + passage_text[2]
        query = 'Q: ' + query
        answer = 'A:' 
   
        

        # 计算 passage 和 query 的长度
        input_ids = []
        passage_ids1 = tokenizer(passage1, add_special_tokens=False)['input_ids']
        passage_ids2 = tokenizer(passage2, add_special_tokens=False)['input_ids']
        passage_ids3 = tokenizer(passage3, add_special_tokens=False)['input_ids']
        query_ids = tokenizer(query, add_special_tokens=False)['input_ids']
        answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        # 检查 \n 被 token 化的方式
        sep_token_id = tokenizer('\n', add_special_tokens=False)['input_ids']
        sep_token_count = len(sep_token_id)  # 计算分隔符 \n 被编码的 token 数
        # 手动拼接 input_ids
        input_ids = passage_ids1 + sep_token_id + passage_ids2 + sep_token_id + passage_ids3 + sep_token_id +query_ids + sep_token_id + answer_ids

        # 将 input_ids 转换为 LongTensor 并加载到 GPU
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions
        last_layer_attention = attentions[-1]

        # 计算 passage 和 query 的长度
        passage_len1 = len(passage_ids1)
        passage_len2 = len(passage_ids2)
        passage_len3 = len(passage_ids3)
        query_len = len(query_ids)

        answer_start_idx = passage_len1 + passage_len2 + passage_len3 + query_len + sep_token_count * 4  # 两个 \n 分隔符

        # 取出 answer 对应的注意力权重
        answer_attention_scores = last_layer_attention[:, :, answer_start_idx:, :]

        passage_attention_scores1 = answer_attention_scores[:, :, :, :passage_len1]
        passage_attention_scores2 = answer_attention_scores[:, :, :, :passage_len1+ sep_token_count:passage_len1 + sep_token_count + passage_len2]
        passage_attention_scores3 = answer_attention_scores[:, :, :, :passage_len1 + sep_token_count + passage_len2+ sep_token_count:passage_len1 + sep_token_count + passage_len2 + sep_token_count + passage_len3]
        query_attention_scores = answer_attention_scores[:, :, :, passage_len1 + sep_token_count + passage_len2 + sep_token_count + passage_len3 + sep_token_count:passage_len1 + sep_token_count + passage_len2 + sep_token_count + passage_len3 + sep_token_count + query_len]

        passage_attention_mean1 = passage_attention_scores1.mean(dim=(1, 2)).mean(dim=-1)  # 对 num_heads 和 answer_len 取平均
        passage_attention_mean2 = passage_attention_scores2.mean(dim=(1, 2)).mean(dim=-1)  # 对 num_heads 和 answer_len 取平均
        passage_attention_mean3 = passage_attention_scores3.mean(dim=(1, 2)).mean(dim=-1)  # 对 num_heads 和 answer_len 取平均
        # 计算对 query 的平均注意力分数
        query_attention_mean = query_attention_scores.mean(dim=(1, 2)).mean(dim=-1)

        psg_attention_means1.append(passage_attention_mean1.cpu())
        psg_attention_means2.append(passage_attention_mean2.cpu())
        psg_attention_means3.append(passage_attention_mean3.cpu())
        query_attention_means.append(query_attention_mean.cpu())

       
    
    # 将所有样本的平均值转换为 Tensor 并计算总体平均值
    psg_attention_means1 = torch.stack(psg_attention_means1)  
    psg_attention_means2 = torch.stack(psg_attention_means2)
    psg_attention_means3 = torch.stack(psg_attention_means3)
    query_attention_means = torch.stack(query_attention_means)  # 将列表转换为 Tensor

    # 对所有样本的 passage 和 query 取平均值
    psg_attention_overall_mean1 = psg_attention_means1.mean(dim=0) 
    psg_attention_overall_mean2 = psg_attention_means2.mean(dim=0) 
    psg_attention_overall_mean3 = psg_attention_means3.mean(dim=0) 
    query_attention_overall_mean = query_attention_means.mean(dim=0)  # 对所有样本求平均

    print("Overall average attention score for passage1:", psg_attention_overall_mean1)
    print("Overall average attention score for passage2:", psg_attention_overall_mean2)
    print("Overall average attention score for passage3:", psg_attention_overall_mean3)
    print("Overall average attention score for query:", query_attention_overall_mean)


    




if __name__ == "__main__":
    main()