
export CUDA_VISIBLE_DEVICES=7
nohup python train_reranker_data.py  \
    --input_path /data/groups/QY_LLM_Other/meisen/rerank_test/train_reranktop10/dev.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /home/lixz23/ragsft/data/marco_v2.1/minicpm_reranker/rerank_10psg_for_dpo/dpo_train  \
    --chat_templete \
    --cut_chunk 8 \
    --number_chunk 7  > chunk7.out  2>&1 &