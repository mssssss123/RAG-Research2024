import argparse
import json
import os

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
                        default="/data/groups/QY_LLM_Other/meisen/radit/dataset/minicpm_r_sft/dpo_train.jsonl")
    parser.add_argument('--output_path', type=str,
                        default="/data/groups/QY_LLM_Other/meisen/radit/dataset/minicpm_r_sft")
    parser.add_argument('--file_name', type=str,
                        default="dpo_train.jsonl")
    args = parser.parse_args()


    file_data = read_jsonl(args.file_path)

    aqua_rat_data = []
    commonsense_qa_data = []
    ecqa_data = []
    gsm8k_data = []
    marco_qa_data = []
    math_qa_data = []
    strategyqa_data = []
    web_questions_data = []
    wiki_qa_data = []
    yahoo_answers_qa_data = []
    for data in file_data:
        data_type = data['data_type']
        if data_type == 'aqua_rat':
            aqua_rat_data.append(data)
        elif data_type == 'commonsense_qa':
            commonsense_qa_data.append(data)
        elif data_type == 'ecqa':
            ecqa_data.append(data)
        elif data_type == 'gsm8k':
            gsm8k_data.append(data)
        elif data_type == 'marcoqa':
            marco_qa_data.append(data)
        elif data_type == 'math_qa':
            math_qa_data.append(data)
        elif data_type == 'strategyqa':
            strategyqa_data.append(data)
        elif data_type == 'web_questions':
            web_questions_data.append(data)
        elif data_type == 'wiki_qa':
            wiki_qa_data.append(data)
        elif data_type == 'yahoo_answers_qa':
            yahoo_answers_qa_data.append(data)
        else:
            print('Warning! Error!')
    aqua_rat_data = aqua_rat_data[:2000]
    commonsense_qa_data = commonsense_qa_data[:2000]
    ecqa_data = ecqa_data[:2000]
    gsm8k_data = gsm8k_data[:2000]
    marco_qa_data = marco_qa_data[:2000]
    math_qa_data = math_qa_data[:2000]
    strategyqa_data = strategyqa_data[:2000]
    web_questions_data = web_questions_data[:2000]
    wiki_qa_data = wiki_qa_data[:2000]
    yahoo_answers_qa_data = yahoo_answers_qa_data[:2000]
    combine_data = []
    combine_data.extend(aqua_rat_data)
    combine_data.extend(commonsense_qa_data)
    combine_data.extend(ecqa_data)
    combine_data.extend(gsm8k_data)
    combine_data.extend(marco_qa_data)
    combine_data.extend(math_qa_data)
    combine_data.extend(strategyqa_data)
    combine_data.extend(web_questions_data)
    combine_data.extend(wiki_qa_data)
    combine_data.extend(yahoo_answers_qa_data)
    print('----')

    output_path = os.path.join(args.output_path, args.file_name)
    with open(output_path, "w") as f:
        for item in combine_data:
            json.dump(item, f)
            f.write("\n")
    print('finish')
















if __name__ == "__main__":
    main()