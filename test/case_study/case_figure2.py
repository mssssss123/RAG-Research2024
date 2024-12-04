
import json
from rouge import Rouge


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data.append(json.loads(line.strip()))
    return data
def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:
        print('Hypothesis is empty.')  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]
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
def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["golds"]:
        if isinstance(item, str):
            ground_truths.add(item)
        elif "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())

    return ground_truths

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

file_path = '/data/groups/QY_LLM_Other/meisen/iclr2024/case/figure2/nqoutput.jsonl'
data = read_jsonl_file(file_path)
true_data = []
for example in data:
    output = example['output']
    answer = get_gold_answers(example)
    score = metric_max_over_ground_truths(output, answer)
    if int(score) == 1:
        true_data.append(example)

write_jsonl(true_data,'/data/groups/QY_LLM_Other/meisen/iclr2024/case/figure2/case.jsonl')
print('---')


