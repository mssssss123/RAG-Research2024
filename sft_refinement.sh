#!/bin/bash
#SBATCH --partition=rag
#SBATCH --nodelist=g1001
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

MODEL_SIZE=8B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=8
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 29502 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /home/meis23/project/ragsft/script/ds_config_zero2.json \
    sft_refinement.py \
    --train_file /data/groups/QY_LLM_Other/meisen/dataset/marco1.0_new/train.jsonl \
    --validation_file /data/groups/QY_LLM_Other/meisen/dataset/marco1.0_new/dev.jsonl \
    --model_name llama \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_passage 5 \
    --max_seq_length 768 \
    --max_psg_length 512 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir  /home/meis23/project/ragsft/checkpoint/iclr/new_refinement \
    --with_tracking \
    --overwrite_cache \
    --report_to tensorboard \
    --logging_steps 1 \
    --checkpointing_steps 500 \
    --eval_steps 500 \
    --seed 2024 \
    --is_instruct \
    --bf16    