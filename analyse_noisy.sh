
export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/llama/trex_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > noisy_llama_rag_trex.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama_new  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/llama/hotpotqa_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > noisy_llama_rag_sft_hotpotqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/llama-8b-sft/LLM-rerank/top_1-5/merge-2200  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/llama/trex_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > noisy_llama_rag_dpo_trex.out  2>&1 &



export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/minicpm/trex_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > noisy_minicpm_rag_trex.out  2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/minicpm/trex_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > noisy_minicpm_rag_sft_trex.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/minicpm-sft/LLM_rerank/top_1-5/merge-2600  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/minicpm/trex_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > noisy_minicpm_rag_dpo_trex.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/minicpm/hotpotqa_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > noisy_minicpm_rag_hotpotqa.out  2>&1 &


export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/minicpm/hotpotqa_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > noisy_minicpm_rag_sft_hotpotqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/minicpm-sft/LLM_rerank/top_1-5/merge-2600  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/noisy/minicpm/hotpotqa_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --user_chat_template  > noisy_minicpm_rag_dpo_hotpotqa.out  2>&1 &