from transformers import T5Tokenizer, T5ForConditionalGeneration, CLIPProcessor, T5Model
from multi_model import MultiModal

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_model(args):
    # clip_model_name = "/data1/zhoutianshuo/pretrain-model/clip-vit-base-patch32"
    clip_model_name=args.clip_model_name
    # t5_model_name = "/data1/zhoutianshuo/pretrain-model/t5-ance"
    t5_model_name=args.t5_model_name
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5Model.from_pretrained(t5_model_name)
    t5_tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]})
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    image_processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = MultiModal(clip_model_name, t5_model, t5_tokenizer)
    return t5_tokenizer, model, image_processor



#
# def load_model(device):
#     clip_model_name = "/data1/zhoutianshuo/pretrain-model/clip-vit-base-patch32"
#     t5_model_name = "/data1/zhoutianshuo/pretrain-model/t5-base"
#     t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
#     t5_model = T5Model.from_pretrained(t5_model_name)
#     t5_tokenizer.add_special_tokens(
#         {"additional_special_tokens": [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]})
#     t5_model.resize_token_embeddings(len(t5_tokenizer))
#     image_processor = CLIPProcessor.from_pretrained(clip_model_name)
#     model = MultiModal(clip_model_name, t5_model, t5_tokenizer)
#     model = model.to(device)
#     return t5_tokenizer, model, image_processor
