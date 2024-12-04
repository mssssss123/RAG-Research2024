import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

pft_model_path = '/home/meis23/project/ra-dit/newcheckpoint/lora_sft/final'
config = PeftConfig.from_pretrained(pft_model_path)
model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
model = PeftModel.from_pretrained(model, pft_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
print('----')
