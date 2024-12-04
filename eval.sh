export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/checkpoint/marco_v2.1/radit/minicpm-sft/minicpm-retriever-no_read/1-3psg_5loop_5tempt/2000_1psg_5loop/merge_final_check  \
    --input_file /home/meis23/project/ragsft/test/trec_match/nq_filter100.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 10  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_top10.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/ragsft/test/trec_match/nq_dev_psg_rerank.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 1  \
    --vllm \
    --user_chat_template  > nq_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/ragsft/test/trec_match/nq_dev_psg_rerank.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 2  \
    --vllm \
    --user_chat_template  > nq_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/ragsft/test/trec_match/nq_dev_psg_rerank.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --user_chat_template  > nq_top5.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/ragsft/test/trec_match/nq_dev_psg_rerank.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 10  \
    --vllm \
    --user_chat_template  > nq_top10.out  2>&1 &


export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 2  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_rerank20_top2.out  2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_rerank20_top5.out  2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 10  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_rerank20_top10.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 1  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_rerank20_top1.out  2>&1 &


export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 2  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_rerank20_top2.out  2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_rerank20_top5.out  2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/nq_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 10  \
    --vllm \
    --rerank \
    --user_chat_template  > nq_rerank20_top10.out  2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/tqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task tqa  \
    --batch_size 2  \
    --top_n 1  \
    --vllm \
    --rerank \
    --user_chat_template  > tqa_rerank20_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/tqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task tqa  \
    --batch_size 2  \
    --top_n 2  \
    --vllm \
    --rerank \
    --user_chat_template  > tqa_rerank20_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/tqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task tqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > tqa_rerank20_top5.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/tqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task tqa  \
    --batch_size 2  \
    --top_n 10  \
    --vllm \
    --rerank \
    --user_chat_template  > tqa_rerank20_top10.out  2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 1  \
    --vllm \
    --rerank \
    --user_chat_template  > fever_rerank20_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 2  \
    --vllm \
    --rerank \
    --user_chat_template  > fever_rerank20_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > fever_rerank20_top5.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 10  \
    --vllm \
    --rerank \
    --user_chat_template  > fever_rerank20_top10.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/marco_qa_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --vllm \
    --rerank \
    --user_chat_template  > marco_rerank20_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/marco_qa_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --vllm \
    --rerank \
    --user_chat_template  > marco_rerank20_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/marco_qa_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 5  \
    --case_num 3000  \
    --vllm \
    --rerank \
    --user_chat_template  > marco_rerank20_top5.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/marco_qa_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 10  \
    --case_num 3000  \
    --vllm \
    --rerank \
    --user_chat_template  > marco_rerank20_top10.out  2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/hotpotqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 1  \
    --vllm \
    --rerank \
    --user_chat_template  > hotpotqa_rerank20_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/hotpotqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 2  \
    --vllm \
    --rerank \
    --user_chat_template  > hotpotqa_rerank20_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragdpo/checkpoint/minicpmwreranker/merge  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/hotpotqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > hotpotqa_rerank20_top5.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/rerank_test/top20/hotpotqa_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 10  \
    --vllm \
    --rerank \
    --user_chat_template  > hotpotqa_rerank20_top10.out  2>&1 &