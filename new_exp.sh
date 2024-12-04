export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/false_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > noanswer_llama_nq.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa/false_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > noanswer_llama_hotpot.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/false_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > noanswer_llama_trex.out  2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/false_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --vllm \
    --rerank \
    --user_chat_template  > noanswer_minicpm_nq.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa/false_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --vllm \
    --rerank \
    --user_chat_template  > noanswer_minicpm_hotpot.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/false_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --vllm \
    --rerank \
    --user_chat_template  > noanswer_minicpm_trex.out  2>&1 &





export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/true_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > hasanswer_llama_nq.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa/true_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > hasanswer_llama_hotpot.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/true_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --vllm \
    --llama_style   \
    --rerank \
    --user_chat_template  > hasanswer_llama_trex.out  2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/nq/true_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task nq  \
    --batch_size 2  \
    --vllm \
    --rerank \
    --user_chat_template  > hasanswer_minicpm_nq.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/hotpopqa/true_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task hotpotqa  \
    --batch_size 2  \
    --vllm \
    --rerank \
    --user_chat_template  > hasanswer_minicpm_hotpot.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /data/groups/QY_LLM_Other/meisen/iclr2024/analyse/knowledegconflict/trex/true_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task t-rex  \
    --batch_size 2  \
    --vllm \
    --rerank \
    --user_chat_template  > hasanswer_minicpm_trex.out  2>&1 &