
export CUDA_VISIBLE_DEVICES=1
nohup python llm_reranker.py  \
    --dataset_file_path /home/lixz23/ragsft/data/marco_v2.1/minicpm_retriever_128_256_top20_test/fever_dev_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/top20  \
    --file_name fever_dev_psg.jsonl  \
    --top_n 20  > fever_dev_psg.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python llm_reranker.py  \
    --dataset_file_path /home/lixz23/ragsft/data/marco_v2.1/minicpm_retriever_128_256_top20_test/hotpotqa_dev_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/top20  \
    --file_name hotpotqa_dev_psg.jsonl  \
    --top_n 20  > hotpotqa_dev_psg.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python llm_reranker.py  \
    --dataset_file_path /home/lixz23/ragsft/data/marco_v2.1/minicpm_retriever_128_256_top20_test/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/top20  \
    --file_name marco_qa_psg.jsonl  \
    --top_n 20  > marco_qa_psg.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python llm_reranker.py  \
    --dataset_file_path /home/lixz23/ragsft/data/marco_v2.1/minicpm_retriever_128_256_top20_test/mmlu_test_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/top20  \
    --file_name mmlu_test_psg.jsonl  \
    --top_n 20  > mmlu_test_psg.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python llm_reranker.py  \
    --dataset_file_path /home/lixz23/ragsft/data/marco_v2.1/minicpm_retriever_128_256_top20_test/nq_dev_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/top20  \
    --file_name nq_dev_psg.jsonl  \
    --top_n 20  > nq_dev_psg.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python llm_reranker.py  \
    --dataset_file_path /home/lixz23/ragsft/data/marco_v2.1/minicpm_retriever_128_256_top20_test/tqa_dev_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/top20  \
    --file_name tqa_dev_psg.jsonl  \
    --top_n 20  > tqa_dev_psg.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python llm_reranker_dpo.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/meisen/radit/dataset/minicpm_r_sft/dev.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/train_reranktop10  \
    --file_name dev.jsonl  \
    --top_n 10  > dev.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python llm_reranker_dpo.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/meisen/radit/dataset/minicpm_r_sft/dpo_train.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/train_reranktop10  \
    --file_name dpo_train.jsonl  \
    --top_n 10  > dpo_train.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python llm_reranker_dpo.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/meisen/radit/dataset/minicpm_r_sft/sft_train.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/newbgererank \
    --output_path /data/groups/QY_LLM_Other/meisen/rerank_test/train_reranktop10  \
    --file_name sft_train.jsonl \
    --top_n 10  > sft_train.out  2>&1 &