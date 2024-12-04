import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

file_path = '/data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl'
ref_nq_data = read_jsonl(file_path)

file_path = '/home/lixz23/rag_instruction/KILT-main/data/nq-dev-kilt.jsonl'
nq_data = read_jsonl(file_path)

file_path = '/home/meis23/project/trec_match/rank_llm-main/rerank_results/BM25/rank_zephyr_7b_v1_full_4096_100_rank_GPT_msmarco-v2.1-doc-segmented.bm25.kilt.nq-dev_2024-07-31T23:23:58.134776_window_20_pass_2.jsonl'
retrieve_data = read_jsonl(file_path)
retrieve_data_dict = {}
for data in retrieve_data:
    qid = data['query']['qid']
    candidates = data['candidates']
    documents = []
    for doc in candidates:
        docid = doc['docid']
        url = doc['doc']['url']
        title = doc['doc']['title']
        headings = doc['doc']['headings']
        segment = doc['doc']['segment']
        start_char = doc['doc']['start_char']
        end_char = doc['doc']['end_char']
        documents.append(
        {
            'docid': docid,
            'url': url,
            'title': title,
            'headings': headings,
            'segment': segment,
            'start_char': start_char,
            'end_char': end_char
        })
    data['documents'] = documents
    retrieve_data_dict[qid] = data

for data in nq_data:
    id = data['id']
    meta = retrieve_data_dict[id]
    passage = meta['documents']
    data['passage'] = passage
# 写入到 .jsonl 文件
file_path = '/home/meis23/project/ragsft/test/trec_match/nq_dev_psg_rerank.jsonl'
with open(file_path, 'w', encoding='utf-8') as f:
    for data in nq_data:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + '\n')

print(f'Data has been written to {file_path}')