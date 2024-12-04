export CUDA_VISIBLE_DEVICES=4
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_fid_nq_lora_sft/step_1700  \
    --input_file /home/meis23/project/FiD-main/open_domain_data/NQ/dev.json \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 50  \
    --metric em  \
    --task fid  \
    --batch_size 4  \
    --top_n 5  \
    --output_path /home/meis23/project/ragsft/result/fid  \
    --exp_name minicpm_dev  \
    --user_chat_template  > minicpm_fid_dev.out  2>&1 &


export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /home/meis23/project/ragsft/checkpoint/minicpm_fid_nq_lora_sft/step_1700  \
    --input_file /home/meis23/project/FiD-main/open_domain_data/NQ/test.json \
    --retrieval_augment \
    --use_lora \
    --max_new_tokens 50  \
    --metric em  \
    --task fid  \
    --batch_size 4  \
    --top_n 5  \
    --output_path /home/meis23/project/ragsft/result/fid  \
    --exp_name minicpm_test  \
    --user_chat_template  > minicpm_fid_test.out  2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/FiD-main/open_domain_data/NQ/test.json \
    --max_new_tokens 50  \
    --metric em  \
    --task fid  \
    --batch_size 4  \
    --output_path /home/meis23/project/ragsft/result/fid/minicpmzero  \
    --exp_name no_rag_test  \
    --user_chat_template  > minicpm_fid_no_rag_test.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/FiD-main/open_domain_data/NQ/dev.json \
    --max_new_tokens 32  \
    --metric em  \
    --task fid  \
    --batch_size 4  \
    --output_path /home/meis23/project/ragsft/result/fid/minicpmzero  \
    --exp_name no_rag_dev  \
    --user_chat_template  > minicpm_fid_no_rag_dev.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/FiD-main/open_domain_data/NQ/test.json \
    --retrieval_augment \
    --max_new_tokens 50  \
    --metric em  \
    --task fid  \
    --batch_size 4  \
    --top_n 5  \
    --output_path /home/meis23/project/ragsft/result/fid/minicpmzero  \
    --exp_name rag_top5_test  \
    --user_chat_template  > minicpm_fid_rag_top5_test.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python eval.py  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16  \
    --input_file /home/meis23/project/FiD-main/open_domain_data/NQ/dev.json \
    --retrieval_augment \
    --max_new_tokens 50  \
    --metric em  \
    --task fid  \
    --batch_size 4  \
    --top_n 5  \
    --output_path /home/meis23/project/ragsft/result/fid/minicpmzero  \
    --exp_name rag_top5_dev  \
    --user_chat_template  > minicpm_fid_rag_top5_dev.out  2>&1 &