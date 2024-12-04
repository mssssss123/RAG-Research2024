
export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/true_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa \
    --exp_name llama_rag_hotpopqa_top5_true \
    --user_chat_template  > llama_rag_nq_top5_true.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/false_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa \
    --exp_name llama_rag_hotpopqa_top5_false \
    --user_chat_template  > llama_rag_nq_top5_false.out  2>&1 &


export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama_new  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/true_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa \
    --exp_name llama_rag_sft_hotpopqa_top5_true \
    --user_chat_template  > llama_rag_sft_nq_top5_true.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama_new  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/false_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa \
    --exp_name llama_rag_sft_hotpopqa_top5_false \
    --user_chat_template  > llama_rag_sft_nq_top5_false.out  2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama_new  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa/true_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > llama_rag_sft_hotpopqa_top5_true.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama_new  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa/false_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > llama_rag_sft_hotpopqa_top5_false.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/llama-8b-sft/LLM-rerank/top_1-5/merge-2200  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/true_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa \
    --exp_name llama_rag_dpo_hotpopqa_top5_true \
    --user_chat_template  > llama_rag_dpo_nq_top5_true.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/llama-8b-sft/LLM-rerank/top_1-5/merge-2200  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/false_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --llama_style   \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa \
    --exp_name llama_rag_dpo_hotpopqa_top5_false \
    --user_chat_template  > llama_rag_dpo_nq_top5_false.out  2>&1 &


export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 4  \
    --vllm \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/minicpm  \
    --exp_name no_rag_minicpm  \
    --user_chat_template  > minicpm_nq_no_rag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 4  \
    --vllm \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa  \
    --exp_name no_rag_minicpm  \
    --user_chat_template  > minicpm_hot_no_rag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 4  \
    --vllm \
    --rerank \
    --llama_style   \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/llama  \
    --exp_name no_rag_llama  \
    --user_chat_template  > llama_nq_no_rag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 4  \
    --vllm \
    --rerank \
    --llama_style   \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/llama  \
    --exp_name no_rag_llama \
    --user_chat_template  > llama_hot_no_rag.out  2>&1 &



export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/true_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --llama_style   \
    --user_chat_template  > llama_rag_trex_true.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/false_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --llama_style   \
    --user_chat_template  > llama_rag_trex_false.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama_new  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/true_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --llama_style   \
    --user_chat_template  > llama_rag_sft_trex_true.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/iclr2024/checkpoint/sft/llama_new  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/false_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --llama_style   \
    --user_chat_template  > llama_rag_sft_trex_false.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/llama-8b-sft/LLM-rerank/top_1-5/merge-2200  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/true_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --llama_style   \
    --user_chat_template  > llama_rag_dpo_trex_true.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/llama-8b-sft/LLM-rerank/top_1-5/merge-2200  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/false_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --top_n 5  \
    --vllm \
    --rerank \
    --llama_style   \
    --user_chat_template  > llama_rag_dpo_trex_false.out  2>&1 &







export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 4  \
    --vllm \
    --rerank \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/minicpm  \
    --exp_name no_rag_minicpm  \
    --user_chat_template  > minicpm_tt_no_rag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 4  \
    --vllm \
    --rerank \
    --llama_style   \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/llama  \
    --exp_name no_rag_llama  \
    --user_chat_template  > llama_tt_no_rag.out  2>&1 &