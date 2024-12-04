import json
import numpy as np
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
def acc_score(prediction, ground_truth):
    if ground_truth in prediction or ground_truth.lower() in prediction or ground_truth.capitalize() in prediction:
        return 1.0
    else:
        return 0.0
def metric_max_over_ground_truths(prediction, ground_truths, fever=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if fever is True and ground_truth in ["REFUTES", "SUPPORTS"]:
            ground_truth = "true" if ground_truth == "SUPPORTS" else "false"
        score = acc_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
def get_gold_answers(output):
    ground_truths = set()
    for item in output:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths
def get_pre_answers(output):
    pre_answer = list()
    for item in output:
        if "text" in item and item["text"] and len(item["text"].strip()) > 0:
            pre_answer.append(item["text"].strip())
    return ' '.join(pre_answer)
result_path = '/home/meis23/project/trec_match/dpo-inference/results/RANK_ZEPHYR_RHO/gpt-4o_8192_20_chatqa_nq_2024-08-11T21:54:34.467243.jsonl'
result_data = read_jsonl(result_path)[:11]

ref_path = '/data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl'
ref_data = read_jsonl(ref_path)
qid_output_dict = {}
for data in ref_data:
    id = data['id']
    qid_output_dict[id] = data['output']

acc_score_list = []
for result in result_data:
    query = result['topic']
    qid = result['topic_id']
    output = qid_output_dict[qid]
    ground_truths = get_gold_answers(output)
    answer = result['answer']
    answer = get_pre_answers(answer)
    score = metric_max_over_ground_truths(answer, ground_truths)
    acc_score_list.append(score)

acc_score_array = np.array(acc_score_list)

# 计算平均值
average_acc_score = np.mean(acc_score_array)

print(f'Average Accuracy Score: {average_acc_score}')
