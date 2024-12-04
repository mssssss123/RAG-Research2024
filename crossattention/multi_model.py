import torch
import torch.nn as nn
from transformers import CLIPModel, T5ForConditionalGeneration, T5Tokenizer, CLIPVisionModel
import numpy as np

class MultiModal(nn.Module):
    def __init__(self, clip_model_name, t5_model, t5_tokenizer):
        super(MultiModal, self).__init__()

        # vision model
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        # language model
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        # 线性层
        self.image_dims = self.clip_model.config.hidden_size
        self.text_dims = self.t5_model.config.hidden_size
        self.projector = nn.Linear(self.image_dims, self.text_dims)
        clip = CLIPModel.from_pretrained(clip_model_name)
        self.logit_scale = clip.logit_scale

        # self.logit_scale=nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.logit_scale_init_value=2.6592
        # self.logit_scale = nn.Parameter(torch.ones([]) * self.logit_scale_init_value)

    def encode_images_only(self, images):

        image_embeddings = self.clip_model(images, output_hidden_states=True)
        # image_embeddings=image_embeddings.last_hidden_state
        # TODO: 可以尝试替换为hidden_state[-2]
        image_embeddings = image_embeddings.last_hidden_state[:, 1:, :]  # batch_size,49,768
        image_embeddings = self.projector(image_embeddings)
        return image_embeddings

    def get_text_inputs_embeds(self, text_inputs, device):
        input_ids = text_inputs["input_ids"].to(device)
        input_embeddings = self.t5_model.get_input_embeddings()
        text_inputs_embeds = input_embeddings(input_ids)
        return text_inputs_embeds

    def get_images_with_caption_inputs_embeds(self, images, img_caps, device):
        # TODO: 修改caption_embedding  ->词向量的embs
        image_embeddings = self.encode_images_only(images)
        img_caps_input_embs = self.get_text_inputs_embeds(img_caps, device)
        merge_input_embs = torch.cat((img_caps_input_embs[:, 0:1, :], image_embeddings, img_caps_input_embs[:, 50:, :]),
                                     dim=1)
        return merge_input_embs

    def get_rep(self, inputs_embeds, input, device):
        if input == None:
            attention_mask = None
        else:
            attention_mask = input['attention_mask'].to(device)
        decoder_input_ids = torch.zeros((inputs_embeds.shape[0], 1), dtype=torch.long)
        decoder_input_ids = decoder_input_ids.to(device)
        outputs = self.t5_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        hidden = outputs.last_hidden_state
        rep = hidden[:, 0, :]
        return rep, hidden

    def get_attention(self, inputs_embeds, input, device):
        if input == None:
            attention_mask = None
        else:
            attention_mask = input['attention_mask'].to(device)
        decoder_input_ids = torch.zeros((inputs_embeds.shape[0], 1), dtype=torch.long)
        decoder_input_ids = decoder_input_ids.to(device)
        outputs = self.t5_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True
        )

        return outputs

    def forward(self, images=None, text_inputs=None, device=None,cross_attention=None, self_attention=None):
            # if attention==None:
            #     if images != None and text_inputs != None:
            #         merge_embs = self.get_images_with_caption_inputs_embeds(images, text_inputs, device)
            #         merge_imgs_rep, _ = self.get_rep(merge_embs, text_inputs, device)
            #         return merge_imgs_rep
            #     elif images != None and text_inputs == None:
            #         image_embs = self.encode_images_only(images)
            #         image_rep, _ = self.get_rep(image_embs, text_inputs, device)
            #         return image_rep
            #     elif images == None and text_inputs != None:
            #         text_embs = self.get_text_inputs_embeds(text_inputs, device)
            #         text_rep, _ = self.get_rep(text_embs, text_inputs, device)
            #         return text_rep
            # else:
            #     raise ValueError("the input is error! ")

        if self_attention:
            merge_embs = self.get_images_with_caption_inputs_embeds(images, text_inputs, device)
            output = self.get_attention(merge_embs, text_inputs, device)
            return output.encoder_attentions


        if cross_attention:
            merge_embs = self.get_images_with_caption_inputs_embeds(images, text_inputs, device)
            output= self.get_attention(merge_embs, text_inputs, device)
            return  output.cross_attentions


