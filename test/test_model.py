import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import safetensors.torch

model_name_or_path = '/home/meis23/project/ra-dit/newcheckpoint/new_re_full_sft/final'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

checkpoint1=safetensors.torch.load_file("/home/meis23/project/ra-dit/newcheckpoint/re_full_sft/final/model-00001-of-00002.safetensors")
checkpoint2=safetensors.torch.load_file("/home/meis23/project/ra-dit/newcheckpoint/re_full_sft/final/model-00002-of-00002.safetensors")

model.load_state_dict(checkpoint1, strict=False)
model.load_state_dict(checkpoint2, strict=False)


print('----')