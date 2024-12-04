import argparse
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
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_file1', type=str, default="/home/meis23/project/ragsft/result/llama/marco/rag_top1/marcooutput.jsonl")
    parser.add_argument('--base_file2', type=str, default="/home/meis23/project/ragsft/result/llama/marco/radit/noread_inst/sft_rag_top1/marcooutput.jsonl")
    parser.add_argument('--output_file', type=str, default="/home/meis23/project/ragsft/new_sft_case_marco_re.jsonl")
    args = parser.parse_args()
    
    no_rag_data = read_jsonl_file(args.base_file1)
    rag_data = read_jsonl_file(args.base_file2)
    saved_list = []
    for no_exp, rag_exp in zip(no_rag_data, rag_data):
        query = no_exp['query']
        answer = no_exp['answers'][0]
        # query = no_exp['input']
        # answer = get_gold_answers(no_exp)
        no_output = no_exp['output']
        rag_output = rag_exp['output']
        no_score = rougel_score(no_output, answer)
        rag_score = rougel_score(rag_output, answer)
        # no_score = metric_max_over_ground_truths(no_output, answer)
        # rag_score = metric_max_over_ground_truths(rag_output, answer)
        no_passage = no_exp['passage'][0]['segment']
        rag_passage = rag_exp['passage'][0]['segment']
        if no_score > rag_score:
            save_data = {
                'query':query,
                'answer':answer,
                # 'answer':list(answer),
                'old_rag_output':no_output,
                'rag_output':rag_output,
                'old_rag_score':no_score,
                'rag_score':rag_score,
                'old_passage':no_passage,
                'rag_passage':rag_passage,
            }
            saved_list.append(save_data)

    with open(args.output_file, 'w') as out_file:
        for entry in saved_list:
            out_file.write(json.dumps(entry, indent=4) + '\n')
    
    print('Saved data to', args.output_file)


















if __name__ == "__main__":
    main()