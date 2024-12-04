#!/bin/bash
#SBATCH --partition=rag
#SBATCH --nodelist=g3013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

MODEL_SIZE=8B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=256
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama pre model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 29501 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /home/meis23/project/ragsft/script/stage3_no_offloading_accelerate.conf \
    sft.py \
    --train_file /data/groups/QY_LLM_Other/meisen/radit/dataset/sft/train_noread.jsonl \
    --validation_file /data/groups/QY_LLM_Other/meisen/radit/dataset/sft/dev_noread.jsonl \
    --model_name llama \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_passage 3 \
    --max_seq_length 1168 \
    --max_psg_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --num_train_epochs 10 \
    --output_dir  /home/meis23/project/ragsft/checkpoint/radit/llama_lora_sft_noread_history \
    --with_tracking \
    --overwrite_cache \
    --report_to tensorboard \
    --logging_steps 1 \
    --checkpointing_steps epoch \
    --eval_steps 100 \
    --seed 2024 \
    --bf16     