import argparse
import json
import os
import random
import jsonlines

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default="/data/groups/QY_LLM_Other/meisen/dataset/marco1.0/train_v2.1.json")
    parser.add_argument('--output_path', type=str,
                        default="/data/groups/QY_LLM_Other/meisen/dataset/marco1.0_new")
   
    args = parser.parse_args()

    input_data = load_file(args.input_path)
    query = input_data['query']
    answers = input_data['answers']
    passages = input_data['passages']
    filter_data = []
    for index in range(len(query)):
        q = query[str(index)]
        ans = answers[str(index)][0]
        psgs = passages[str(index)]
        if ans == 'No Answer Present.':
            continue
        filter_data.append(
            {
                'query': q,
                'passages': psgs,
            }
        )
    sampled_data = random.sample(filter_data, 32000)
    sample_train_data = sampled_data[:30000]
    sample_dev_data = sampled_data[30000:]

    new_train_data = []
    for data in sample_train_data:
        query = data['query']
        passages = data['passages']
        true_psg = []
        false_psg = []
        for psg in passages:
            if psg['is_selected'] == 1:
                true_psg.append(psg)
            else:
                false_psg.append(psg)
        if len(true_psg) == 0:
            continue
        true_d = random.sample(true_psg,1)
        false_d = random.sample(false_psg,1)
        new_train_data.append(
                    {
                        'query': query,
                        'passages': true_d,
                        'label':'YES',
                    }
                )
        new_train_data.append(
                    {
                        'query': query,
                        'passages': false_d,
                        'label':'NO',
                    }
                )
    
    new_dev_data = []
    for data in sample_dev_data:
        query = data['query']
        passages = data['passages']
        true_psg = []
        false_psg = []
        for psg in passages:
            if psg['is_selected'] == 1:
                true_psg.append(psg)
            else:
                false_psg.append(psg)
        if len(true_psg) == 0:
            continue
        false_d = random.sample(false_psg,1)
        true_d = random.sample(true_psg,1)
        new_dev_data.append(
                    {
                        'query': query,
                        'passages': true_d,
                        'label':'YES',
                    }
                )
        new_dev_data.append(
                    {
                        'query': query,
                        'passages': false_d,
                        'label':'NO',
                    }
                )
    print('---')

    output_path = os.path.join(args.output_path, 'train.jsonl')
    with open(output_path, "w") as f:
        for item in new_train_data:
            json.dump(item, f)
            f.write("\n")
    print('finish1')




    output_path1 = os.path.join(args.output_path, 'dev.jsonl')
    with open(output_path1, "w") as f:
        for item in new_dev_data:
            json.dump(item, f)
            f.write("\n")
    print('finish2')











if __name__ == "__main__":
    main()