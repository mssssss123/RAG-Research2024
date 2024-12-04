export CUDA_VISIBLE_DEVICES=0
nohup python cal_attention_score.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --top_n 5   > minicpm_rag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python cal_attention_score.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/minicpm  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --top_n 5   > minicpm_rag_sft.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python cal_attention_score.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/minicpm-sft/LLM_rerank/top_1-5/merge-2600  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --top_n 5   > minicpm_rag_dpo.out  2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python cal_attention_score.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --top_n 5   > llama_rag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python cal_attention_score.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --top_n 5   > llama_rag_sft.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python cal_attention_score.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/llama-8b-sft/LLM-rerank/top_1-5/merge-2200  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --top_n 5   > llama_rag_dpo.out  2>&1 &