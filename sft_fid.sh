#!/bin/bash
#SBATCH --partition=rag
#SBATCH --nodelist=g3013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

MODEL_SIZE=400M
NUM_GPUS=4
BATCH_SIZE_PER_GPU=64
TOTAL_BATCH_SIZE=256
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training fid model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --main_process_port 29501 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    train_fid.py \
    --train_file /home/meis23/project/FiD-main/open_domain_data/NQ/new_train.json \
    --validation_file /home/meis23/project/FiD-main/open_domain_data/NQ/dev.json \
    --model_name_or_path /home/meis23/project/pretrained_model/t5/t5-base \
    --num_passage 5 \
    --text_maxlength 250 \
    --answer_maxlength 20 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --num_train_epochs 20 \
    --output_dir /home/meis23/project/ragsft/checkpoint/nqfid \
    --with_tracking \
    --overwrite_cache \
    --report_to tensorboard \
    --logging_steps 1 \
    --checkpointing_steps 100 \
    --eval_steps 100 \
    --seed 2024 \
    --use_checkpoint     