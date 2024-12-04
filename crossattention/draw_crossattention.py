from sklearn.manifold import TSNE
from transformers import T5Tokenizer, T5ForConditionalGeneration,RobertaTokenizer
import json
import json
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics.pairwise import euclidean_distances
from statistics import mean
from sklearn.decomposition import PCA
from utils import load_model
from visual import TSVFile
from PIL import Image
import os
import base64
import io
from transformers import T5Tokenizer, T5ForConditionalGeneration,RobertaTokenizer,T5Model
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, ticker
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
import statistics
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_imag(index_file_path=None, image_path=None):
    img_map = {}
    all_img_num = 0
    with open(index_file_path) as fin:
        for line in fin:
            tokens = line.strip().split('\t')
            all_img_num += 1
            img_map[tokens[0]] = int(tokens[1])
    img_tsv = TSVFile(image_path, all_img_num)
    return img_tsv,img_map
def load_data(file_path=None):
    data = []
    with open(file_path) as f:
        for line in tqdm(f):
            line = line.strip()
            js = json.loads(line)
            data.append(js)
    return data

def load_docs(path):
    data = {}
    with open(path) as fin:
        for line in tqdm(fin):
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            data[did] = example['fact']
    return data

def load_caps(path):
    data = {}
    with open(path) as fin:
        for line in tqdm(fin):
            example = json.loads(line.strip())
            imgid = str(example['image_id'])
            data[imgid] = example['caption']
    return data
def get_example(data,docs,cap,img_tsv,img_map):

    pos_img_list=[]
    pos_text_list=[]

    for idx, item in enumerate(tqdm(data)):
        if item['img_posFacts']!=[] and len(pos_img_list)<100:
            group={}
            group["img_posFacts"] = item['img_posFacts']
            group["qid"] = item['qid']
            group["query"] = item['Q']
            pos_img_list.append(group)

        if item['txt_posFacts']!=[] and len(pos_text_list)<100:
            group={}
            group["txt_posFacts"] = item['txt_posFacts']
            group["qid"] = item['qid']
            group["query"] = item['Q']
            pos_text_list.append(group)

        if len(pos_text_list)+len(pos_img_list)>200:
            break

    for img in pos_img_list:
        imag_idx = random.choice(img['img_posFacts'])
        offset = img_map[imag_idx]
        tsv_img = img_tsv[offset]
        img["pos_idx"] = tsv_img[0]
        img["imag_text"] = tsv_img[1]
        img["cap_text"] = cap[imag_idx]
        print("--------------")

    for text in pos_text_list:
        text_idx = random.choice(text['txt_posFacts'])
        text["doc"] = docs[text_idx]




    print("--------------")

    return pos_text_list, pos_img_list
def get_cross_attention(model,tokenizer,image_processor,pos_text_list,pos_img_list,device):
    img_special_len=49
    pre_token= DEFAULT_IM_START_TOKEN+" "+ DEFAULT_IMAGE_PATCH_TOKEN * img_special_len + DEFAULT_IM_END_TOKEN
    query_img=[]
    text_len=128
    cap_len=51+51
    img_all=[]
    cap_all=[]
    for line in pos_img_list:
        query=line['query']
        query_img.append(query)
        img=line['imag_text']
        img = image_processor(images=Image.open(io.BytesIO(base64.b64decode(img))), return_tensors="pt")["pixel_values"][0]
        img_all.append(img)
        cap=pre_token+" "+"caption: "+line['cap_text']
        cap_all.append(cap)
    assert len(img_all)==len(cap_all)
    query_img=tokenizer(query_img, return_tensors='pt',max_length=text_len,padding='max_length',truncation=True).to(device)

    cap_all=tokenizer(cap_all,return_tensors='pt',max_length=cap_len,padding='max_length',
                      truncation=True,add_special_tokens=False).to(device)
    img_all=torch.stack(img_all, dim=0).to(device)
    text = cap_all

    # text = cap_all[0]
    # cap_all = tokenizer(cap_all[0], return_tensors='pt',truncation=True,add_special_tokens=False ).to(device)
    # img_all =img_all[0]
    # img_all=img_all.unsqueeze(0).to(device)

    query_txt=[]
    txt_all=[]
    for line in pos_text_list:
        query=line['query']
        query_txt.append(query)
        txt=line['doc']
        txt_all.append(txt)
    query_txt=tokenizer(query_txt, return_tensors='pt',max_length=text_len,padding='max_length',truncation=True).to(device)
    txt_all=tokenizer(txt_all,return_tensors='pt',max_length=cap_len,padding='max_length',truncation=True).to(device)


    v=model(img_all,cap_all,device,True)
    attentions = torch.zeros_like(v[0])
    for layer in v:
        attentions += layer

    heads = attentions.shape[1]
    attention_mean = torch.mean(attentions, dim=1)
    singel=False
    if singel:
        attention_mean = attention_mean[0, :, :]
    else:
        attention_mean = attention_mean.squeeze(1)

    lsm = F.softmax(attention_mean, dim=1)

    lsm = lsm.detach().cpu().numpy()
    print("-----------")
    return lsm,text

def get_self_attention(model,tokenizer,image_processor,pos_text_list,pos_img_list,device):
    img_special_len=49
    pre_token= DEFAULT_IM_START_TOKEN+" "+ DEFAULT_IMAGE_PATCH_TOKEN * img_special_len + DEFAULT_IM_END_TOKEN
    query_img=[]
    text_len=128
    cap_len=51+128
    img_all=[]
    cap_all=[]
    for line in pos_img_list:
        query=line['query']
        query_img.append(query)
        img=line['imag_text']
        img = image_processor(images=Image.open(io.BytesIO(base64.b64decode(img))), return_tensors="pt")["pixel_values"][0]
        img_all.append(img)
        cap=pre_token+" "+"caption: "+line['cap_text']
        cap_all.append(cap)
    assert len(img_all)==len(cap_all)
    query_img=tokenizer(query_img, return_tensors='pt',max_length=text_len,padding='max_length',truncation=True).to(device)
    img_all = torch.stack(img_all, dim=0).to(device)

    # text = cap_all[0]
    # cap_all = tokenizer(cap_all[0], return_tensors='pt',truncation=True,add_special_tokens=False ).to(device)
    # img_all =img_all[0]
    # img_all=img_all.unsqueeze(0).to(device)

    query_txt=[]
    txt_all=[]
    for line in pos_text_list:
        query=line['query']
        query_txt.append(query)
        txt=line['doc']
        txt_all.append(txt)


    singel=True

    if singel:
        all_picture_attention=[]
        all_caption_attention=[]
        for idx in range(len(cap_all)):
            cap_one = cap_all[idx]
            img_one = img_all[:1, ...]

            input_cap_one = tokenizer(cap_one, return_tensors='pt', truncation=True, add_special_tokens=False).to(
                device)

            text = cap_all
            v = model(img_one, input_cap_one, device, True, None)
            attentions = torch.zeros_like(v[0])
            for layer in v:
                attentions += layer

            heads = attentions.shape[1]
            attention_mean = torch.mean(attentions, dim=1)
            # select one attention
            attention_mean = attention_mean[0, :, :]

            cap_token_ids = tokenizer(cap_one).input_ids
            text = tokenizer.convert_ids_to_tokens(cap_token_ids)
            text = [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in text]

            lsm = F.softmax(attention_mean, dim=1)

            lsm = lsm.detach().cpu().numpy()
            mean_picture = np.mean(lsm[:, :51], axis=1).tolist()
            mean_caption = np.mean(lsm[:, 51:], axis=1).tolist()

            all_picture_attention.append(mean_picture[0])
            all_caption_attention.append(mean_caption[0])



        average_picture = statistics.mean(all_picture_attention)
        average_caption = statistics.mean(all_caption_attention)

        ratiao_picture = average_picture/(average_picture + average_caption)
        ratiao_caption =  average_caption/(average_picture + average_caption)



        print("-----------")




    else:
        cap_all = tokenizer(cap_all, return_tensors='pt', max_length=cap_len, padding='max_length',
                            truncation=True, add_special_tokens=False).to(device)

        text = cap_all
        v = model(img_all, cap_all, device, None, True)
        attentions = torch.zeros_like(v[0])
        for layer in v:
            attentions += layer

        heads = attentions.shape[1]
        attention_mean = torch.mean(attentions, dim=1)
        # select one attention
        attention_mean = attention_mean[2, :, :]

        lsm = F.softmax(attention_mean, dim=1)

        lsm = lsm.detach().cpu().numpy()
        print("-----------")
        return lsm, text






def get_sub_attention(lsm,text, x_start, y_start,x_end, y_end):
    sub_lsm = lsm[y_start:y_end,x_start:x_end]
    text = text[x_start:]
    return sub_lsm, text

def encoder(model,tokenizer,code):
    pos = tokenizer(code, add_special_tokens=False, return_tensors='pt', truncation=True)
    code_inputs = pos.input_ids
    code_masks = pos.attention_mask

    decoder_input_ids = torch.zeros((code_inputs.shape[0], 1), dtype=torch.long)
    code_out = model(
        input_ids=code_inputs,
        attention_mask=code_masks,
        output_attentions=True,
        decoder_input_ids=decoder_input_ids)
    keys = code_out.keys

    ids = code_inputs.tolist()[0]
    text = tokenizer.convert_ids_to_tokens(ids)
    text = [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in text]

    v = code_out.cross_attentions
    attentions = torch.zeros_like(v[0])
    for layer in v:
        attentions += layer

    heads = attentions.shape[1]
    attention_mean = torch.mean(attentions, dim=1)
    attention_mean = attention_mean[0, :, :]
    lsm = F.softmax(attention_mean, dim=1)

    lsm = lsm.detach().numpy()
    return lsm,text

def draw_picture(scores,text,output_file):
    variables = text
    labels = ['CodeT5','CodeT5 (w.STA)','TESDR']

    df = pd.DataFrame(scores, columns=variables, index=labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='Blues')
    fig.colorbar(cax,shrink=0.8)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xticklabels([''] + list(df.columns),fontsize=1)
    ax.set_yticklabels([''] + list(df.index),fontsize=15)
    #plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    print("-------")

def attention_plot(attention, x_texts, y_texts, figsize, output,annot=False,figure_path='./figures',
                   figure_name='attention_weight.png'):
    mean_picture = np.mean(attention[:, :51], axis=1)
    mean_caption = np.mean(attention[:, 51:], axis=1)

    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=5)
    # heatmap
    hm = sns.heatmap(attention,
                     cbar=False,
                     # cmap="RdBu_r",
                     cmap='Blues',
                     # annot=annot,
                     square=True,
                     # fmt='.2f',
                     # annot_kws={'size': 10},
                     #yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    plt.xticks(fontsize=20)
    #plt.yticks(fontsize=50)
    label_x=ax.get_xticklabels()
    plt.setp(label_x,rotation=90)
    #label_y=ax.get_yticklabels()
    #plt.setp(label_y, rotation=90)
    plt.savefig(output)
    plt.show()



    print("---------------------------")


def main():
    parser = ArgumentParser()

    parser.add_argument('--input', type=str,
                        default="/data1/zhoutianshuo/multi-modal/webqa_data/test.json")
    parser.add_argument('--output_image', type=str,
                        default="./image.png")

    parser.add_argument('--output_text', type=str,
                        default="./text.png")

    best_train_clip = "/data1/zhoutianshuo/multi-modal/UniVL-T5/DR/inbatch_hard_neg/checkpoint_multi_hn_t5_ance_cos_hard_batch_neg_new_len_way2_new/model.best.pt"
    best="/data1/zhoutianshuo/multi-modal/UniVL-T5/DR/inbatch_hard_neg/checkpoint_multi_hn_t5_ance_cos_hard_batch_neg_fix_clip_new_len_no_extend/model.best.pt"


    first_step_fix_t5 ="/data1/zhoutianshuo/multi-modal/UniVL-T5/DPR/checkpoint_multi_inb_t5_ance_cos_fix_t5_new_len/model.best.pt"
    first_step_fix_clip="/data1/zhoutianshuo/multi-modal/UniVL-T5/DPR/checkpoint_multi_inb_t5_ance_cos_fix_clip_new_len/model.best.pt"
    first_step_fix_t5_and_clip="/data1/zhoutianshuo/multi-modal/UniVL-T5/DPR/checkpoint_multi_inb_t5_ance_cos_only_finetuned_projector/model.best.pt"
    first_step_all_finetune="/data1/zhoutianshuo/multi-modal/UniVL-T5/DPR/checkpoint_multi_inb_t5_ance_cos_new_len/model.best.pt"

    parser.add_argument("--doc_path", type=str,default='/data1/zhoutianshuo/multi-modal/webqa_data/all_docs.json')
    parser.add_argument("--cap_path", type=str,default='/data1/zhoutianshuo/multi-modal/webqa_data/all_imgs_query.json') #all_imgs_query all_imgs
    parser.add_argument("--img_feat_path", type=str,default='/data1/zhoutianshuo/multi-modal/webqa_data/imgs.tsv')
    parser.add_argument("--img_linelist_path", type=str,default='/data1/zhoutianshuo/multi-modal/webqa_data/imgs.lineidx.new')
    parser.add_argument('--checkpoint',type=str,default='/data1/zhoutianshuo/multi-modal/UniVL-T5/DR/inbatch_hard_neg/checkpoint_multi_hn_t5_ance_cos_hard_batch_neg_new_len_way2_new/model.best.pt')
    parser.add_argument('--t5_model_name',type=str,default='/data1/zhoutianshuo/pretrain-model/t5-ance')
    parser.add_argument('--clip_model_name',type=str,default = "/data1/zhoutianshuo/pretrain-model/clip-vit-base-patch32")

    args = parser.parse_args()
    args.checkpoint =  first_step_fix_t5_and_clip

    data = load_data(args.input)
    docs = load_docs(args.doc_path)
    cap = load_caps(args.cap_path)
    img_tsv, img_map = load_imag(args.img_linelist_path, args.img_feat_path)


    tokenizer, model, image_processor=load_model(args)
    if args.checkpoint!=None:
        model.load_state_dict(torch.load(args.checkpoint)['model'])

    random.seed(20000)#20000

    random.shuffle(data)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    pos_text_list, pos_img_list=get_example(data,docs,cap,img_tsv,img_map)

    #lsm,text = get_cross_attention(model, tokenizer, image_processor, pos_text_list, pos_img_list, device)
    lsm, text = get_self_attention(model, tokenizer, image_processor, pos_text_list, pos_img_list, device)
    lsm, text = get_sub_attention(lsm, text,x_start=53, y_start=0,x_end=1000, y_end=51)



    attention_plot(lsm, x_texts=text, y_texts=['attention'], annot=False, figsize=(80, 20),
                   output='/data1/zhoutianshuo/multi-modal/test/MNID/first_step_all_finetune',
                   figure_path='./figures_test',
                   figure_name='t5_decoder_attention_weight_mean__softmax_src_target.pdf')
    print("--------------")




if __name__ == '__main__':
    main()

