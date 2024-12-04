import json

import jsonlines

def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst
def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["output"]:
        if isinstance(item, str):
            ground_truths.add(item)
        elif "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())

    return ground_truths
def has_an(prediction, ground_truths):
    scores_for_ground_truths =[]
    for ground_truth in ground_truths:
        score = _acc_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
def _acc_score(prediction, ground_truth):
    if ground_truth in prediction or ground_truth.lower() in prediction or ground_truth.capitalize() in prediction:
        return 1.0
    else:
        return 0.0
    
data = load_file("/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/llama_merge_2200_reward/merge_final_random_100/nq_list/ne_dev_psg.jsonl")
total_passage_num = 5 * len(data)
gold_psg_num = 0
for example in data:
    if True not in example['judge_preds']:
        continue
    gold_candidate_answers = get_gold_answers(example)
    passages = example['rerank_passage'][:5]
    for passage in passages:
        prediction = str(passage['segment']).strip()
        if has_an(prediction, gold_candidate_answers):
            gold_psg_num += 1

ratio = gold_psg_num / total_passage_num
print('Ratio: {:.4f}'.format(ratio))

