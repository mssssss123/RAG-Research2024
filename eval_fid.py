import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,T5ForConditionalGeneration,T5Tokenizer
from baselines.fid.evaluation import ems
from utils.eval_utils import load_file, postprocess_output, test_kilt_em, match
CHOICE_TASK =['arc','hellaswag','socialiqa','piqa',]
KILT_TASK = ['fever','aida','t-rex','eli5','hotpotqa','wow','nq','marco']

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np

class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        # outputs.last_hidden_state = outputs[0].view(bsz, self.n_passages *
        #                                             passage_length, -1)
        outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output


def get_target(example):
    if 'target' in example:
        target = example['target']
        return target + ' </s>'
    elif 'answers' in example:
        return random.choice(example['answers']) + ' </s>'
    else:
        return None


def process_input_data(input_data, args):
    new_input_data = []
    for example in input_data:
        question = "question:" + " " + example['question']
        target = get_target(example)
        if 'ctxs' in example and args.top_n is not None:
            f = "title:" + " {} " + "context:" + " {}"
            if 'score' in example['ctxs'][0]:
                example['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)
            contexts = example['ctxs'][:args.top_n]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages = None
        new_input_data.append(
            {
                'question' : question,
                'target' : target,
                'passages' : passages,
                'gt':example['answers'],
            }
        )
  
    return new_input_data

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


def call_model(args, prompts, user_chat_template, model, tokenizer, max_new_tokens=100):

    if user_chat_template:
        chat_prompts = []
        for prompt in prompts:
            if args.llama_style:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = "<用户>{}<AI>".format(prompt)
            chat_prompts.append(prompt)
        prompts = chat_prompts
        
        
    input_ids,attention_mask = encode_passages(prompts,tokenizer,512)
    outputs = model.generate(input_ids=input_ids.cuda(),
                                attention_mask=attention_mask.cuda(),
                                max_length=max_new_tokens
                                )

    text_list = []
    for ids, tokens in enumerate(input_ids):
        text = tokenizer.decode(outputs[ids], skip_special_tokens=True)
        # actual_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        text_list.append(text)

    preds = []
    for pred in text_list:
        pred = pred.lstrip()
        preds.append(pred.split("\n")[0])

    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default='/home/meis23/project/ragsft/checkpoint/nqfid/step_2700')
    parser.add_argument('--input_file', type=str, default='/home/meis23/project/FiD-main/open_domain_data/NQ/dev.json')
    parser.add_argument('--retrieval_augment', action='store_true',default= True)

    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--top_n', type=int, default=5,help="number of paragraphs to be considered.")

    parser.add_argument('--batch_size', type=int, default=256)
  
    # 用于case_study
    parser.add_argument('--output_path', type=str,
                        default='/home/meis23/project/ragsft/result/fid')
    parser.add_argument('--exp_name', type=str, default='t5-dev')
    parser.add_argument('--case_num', type=int, default=-1)
    args = parser.parse_args()

    if args.output_path!=None:
        output_path = os.path.join(args.output_path, args.exp_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path=None

    tokenizer = T5Tokenizer.from_pretrained('/data/groups/QY_LLM_Other/lixinze/pretrain_model/t5-base', return_dict=False)
    model_class = FiDT5
    model = model_class.from_pretrained(args.model_name_or_path)

    model.eval().cuda()
    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]

    input_data = process_input_data(input_data, args)
    exactmatch = []
    total = 0
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        predict_text = []
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        with torch.no_grad():
            context_ids, context_mask = encode_passages(text_passages,
                                                        tokenizer,
                                                        max_length=200)
            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = batch[k]
                if 'gt' in example:
                    score = ems(ans, example['gt'])
                    exactmatch.append(score)
                    predict_text.append(ans)
                total += 1
               
        for j, item in enumerate(batch):
            pred = predict_text[j]
            item["output"] = pred
        
    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx + 1) * args.batch_size:]
        predict_text = []
        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        with torch.no_grad():
            context_ids, context_mask = encode_passages(text_passages,
                                                        tokenizer,
                                                        max_length=200)
            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = batch[k]
                if 'gt' in example:
                    score = ems(ans, example['gt'])
                    exactmatch.append(score)
                    predict_text.append(ans)
                total += 1
               
        for j, item in enumerate(batch):
            pred = predict_text[j]
            item["output"] = pred

    if output_path is not None:
        output_path = os.path.join(output_path, 'output.jsonl')
        with open(output_path, "w") as f:
            for item in input_data:
                json.dump(item, f)
                f.write("\n")
    print("结果文件已生成：", output_path)
    score, total = np.mean(exactmatch), total


    print("overall result: {0}".format(
        score))
    print('finish')


if __name__ == "__main__":
    main()