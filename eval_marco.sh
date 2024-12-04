## minicpm wo rag marco

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history  \
    --exp_name no_rag  \
    --user_chat_template  > minicpm_marco_norag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history/notemp  \
    --exp_name no_rag  > minicpm_marco_norag_notemp.out  2>&1 &



## minicpm rag marco

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 1  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history  \
    --exp_name rag_top1  \
    --user_chat_template  > minicpm_marco_rag_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 2  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history  \
    --exp_name rag_top2  \
    --user_chat_template  > minicpm_marco_rag_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/minicpm-history-test  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 3  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history  \
    --exp_name rag_top3  \
    --user_chat_template  > minicpm_marco_rag_top3.out  2>&1 &

## minicpm sft rag marco

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 3  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > minicpm_marco_sft_rag_top3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 2  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > minicpm_marco_sft_rag_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 1  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > minicpm_marco_sft_rag_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 3  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > minicpm_marco_sft_rag_top3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 2  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > minicpm_marco_sft_rag_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/minicpm_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --top_n 1  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/minicpm/marco/radit/history  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > minicpm_marco_sft_rag_top1.out  2>&1 &

## llama wo rag marco

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco  \
    --exp_name no_rag  \
    --user_chat_template  > llama_marco_norag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/llama/marco/pretrain  \
    --exp_name no_rag  > llama_marco_norag.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 4  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/chatqa  \
    --exp_name no_rag  \
    --user_chat_template  > llama_marco_norag_chatqa.out  2>&1 &

## llama rag marco


export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco  \
    --exp_name rag_top1  \
    --user_chat_template  > llama_marco_rag_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco  \
    --exp_name rag_top2  \
    --user_chat_template  > llama_marco_rag_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 3  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco  \
    --exp_name rag_top3  \
    --user_chat_template  > llama_marco_rag_top3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/llama/marco/pretrain  \
    --exp_name rag_top1   > llama_marco_rag_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/llama/marco/pretrain  \
    --exp_name rag_top2  > llama_marco_rag_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 3  \
    --case_num 3000  \
    --output_path /home/meis23/project/ragsft/result/llama/marco/pretrain  \
    --exp_name rag_top3  > llama_marco_rag_top3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/chatqa  \
    --exp_name rag_top1  \
    --user_chat_template  > llama_marco_rag_top1_chatqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/chatqa  \
    --exp_name rag_top2  \
    --user_chat_template  > llama_marco_rag_top2_chatqa.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Llama3-ChatQA-1.5-8B  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 3  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/chatqa  \
    --exp_name rag_top3  \
    --user_chat_template  > llama_marco_rag_top3_chatqa.out  2>&1 &

## llama sft rag marco

export CUDA_VISIBLE_DEVICES=7
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/llama_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > llama_marco_sft_rag_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/llama_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > llama_marco_sft_rag_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/llama_marco_lora_sft/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 3  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > llama_marco_sft_rag_top3.out  2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 3  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/radit/noread  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > llama_marco_sft_rag_top3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/radit/noread  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > llama_marco_sft_rag_top2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/radit/noread  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > llama_marco_sft_rag_top1.out  2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_instruct_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 3  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/radit/noread_inst  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > llama_marco_sft_rag_top3_inst.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_instruct_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/radit/noread_inst  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > llama_marco_sft_rag_top2_inst.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_instruct_lora_sft_noread_history/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/radit/noread_inst  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > llama_marco_sft_rag_top1_inst.out  2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_chatqa_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 1  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/chatqa  \
    --exp_name sft_rag_top1  \
    --user_chat_template  > llama_marco_sft_rag_top1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_chatqa_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 2  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/chatqa  \
    --exp_name sft_rag_top2  \
    --user_chat_template  > llama_marco_sft_rag_top2.out  2>&1 &


export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/radit/llama_chatqa_lora_sft_noread/epoch_0  \
    --input_file /home/lixz23/ragsft/data/marco_v2.1/test_data/dev_v2.1_psg.jsonl \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 100  \
    --metric rouge  \
    --task marco  \
    --batch_size 2  \
    --top_n 3  \
    --case_num 3000  \
    --llama_style   \
    --output_path /home/meis23/project/ragsft/result/llama/marco/chatqa  \
    --exp_name sft_rag_top3  \
    --user_chat_template  > llama_marco_sft_rag_top3.out  2>&1 &