from baselines.fid.dataset import Collator, preprocess_function, preprocess_function_nq
from baselines.fid.model import FiDT5
from baselines.fid.train_args import debug_parse_args, parse_args

import logging
import math
import os
import random
import datasets
import torch
import copy
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Dict, Sequence
import json
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    DataCollatorForSeq2Seq,
)
from datetime import timedelta
from utils.train_utils import save_checkpoint


logger = get_logger(__name__)

def main():

   
    args = parse_args()
    # args = debug_parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[kwargs],
        **accelerator_log_kwargs,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()


    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
    if extension == "jsonl" or extension == "jsonlines":
        extension = "json"
    raw_datasets = load_dataset(extension, data_files=data_files)

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name_or_path)

    t5 = transformers.T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    model.set_checkpoint(args.use_checkpoint)
  
    
    column_names = raw_datasets["train"].column_names
   
    encode_function = partial(
        preprocess_function_nq,
        num_passage=args.num_passage,
    )

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on training dataset",
        )

        eval_dataset = raw_datasets["validation"].map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on evaluating dataset",
        )
    
    data_collator = Collator(args.text_maxlength, tokenizer, answer_maxlength=args.answer_maxlength)
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)
    
    # Metric
    # 如果评估时需要使用评价指标在这里加

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
  
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                (labels, _, context_ids, context_mask) = batch
                loss = model(
                    input_ids=context_ids,
                    attention_mask=context_mask,
                    labels=labels
                )[0]
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                      
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        accelerator.wait_for_everyone()
                        save_checkpoint(model, tokenizer, accelerator, args, output_dir)

                if completed_steps % args.eval_steps == 0:
                    model.eval()
                    total_eval_loss = []
                    for step, batch in enumerate(eval_dataloader): 
                        (labels, _, context_ids, context_mask) = batch
                        with torch.no_grad():
                            eval_loss = model(
                                input_ids=context_ids,
                                attention_mask=context_mask,
                                labels=labels
                            )[0]
                        total_eval_loss.append(accelerator.gather_for_metrics(eval_loss.repeat(args.per_device_eval_batch_size)))
                    total_eval_loss = torch.cat(total_eval_loss)
                    avg_eval_loss = torch.mean(total_eval_loss)
                    logger.info(f"Step: {completed_steps}, Dev Loss: {avg_eval_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "eval_loss": avg_eval_loss,
                            },
                            step=completed_steps,
                        )
                    model.train()

                        
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
    
            
            if completed_steps >= args.max_train_steps:
                break
        
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            accelerator.wait_for_everyone()
            save_checkpoint(model, tokenizer, accelerator, args, output_dir)
  

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        output_dir = f"final"
        output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.wait_for_everyone()
        save_checkpoint(model, tokenizer, accelerator, args, output_dir)
        
     
    
    print('finish')
    




if __name__ == "__main__":
    main()