export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 4  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/fever/chatqa  \
    --exp_name no_rag  \
    --user_chat_template  > llama_fever_norag_chatqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/fever/chatqa  \
    --exp_name rag_top3  \
    --user_chat_template  > llama_fever_rag_top3_chatqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 2  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/fever/chatqa  \
    --exp_name rag_top2  \
    --user_chat_template  > llama_fever_rag_top2_chatqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 1  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/fever/chatqa  \
    --exp_name rag_top1  \
    --user_chat_template  > llama_fever_rag_top1_chatqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_chatqa_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/fever/chatqa  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > llama_fever_sft_rag_top3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_chatqa_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 1  \
    --top_n 2  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/fever/chatqa  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > llama_fever_sft_rag_top2.out  2>&1 &


export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_chatqa_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 1  \
    --top_n 1  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/fever/chatqa  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > llama_fever_sft_rag_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 32  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever  \
    --exp_name no_rag  \
    --user_chat_template  > fever.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 32  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/history  \
    --exp_name no_rag  \
    --user_chat_template  > fever_history.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 32  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/history/notemp  \
    --exp_name no_rag  > fever_history_notemp.out  2>&1 &


export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 2  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/history  \
    --exp_name rag_top2  \
    --user_chat_template  > fever_rag2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/history  \
    --exp_name rag_top3  \
    --user_chat_template  > fever_rag3_history.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit  \
    --exp_name rag_top3  \
    --user_chat_template  > fever_rag3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/history  \
    --exp_name rag_top1  \
    --user_chat_template  > fever_rag1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > fever_sft_rag1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 2  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > fever_sft_rag2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > fever_sft_rag3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > fever_radit_sft_rag1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 2  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > fever_radit_sft_rag2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > fever_radit_sft_rag3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/new  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > fever_radit_sft_rag1_no.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 2  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/new  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > fever_radit_sft_rag2_no.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/new  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > fever_radit_sft_rag3_no.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 1  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/history  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > fever_radit_sft_rag1_history.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 2  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/history  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > fever_radit_sft_rag2_history.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/fever_dev_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 32  \
    --metric accuracy  \
    --task fever  \
    --batch_size 2  \
    --top_n 3  \
    --output_path /home/meis23/project/ragsft/result/minicpm/fever/radit/history  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > fever_radit_sft_rag3_history.out  2>&1 &