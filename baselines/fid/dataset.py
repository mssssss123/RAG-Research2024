import random
import torch


def preprocess_function(example):
    question = "question:" + " " + example['query']
    answer = example['answers']
    if isinstance(answer, list):
        answer = answer[0]
    target = answer + ' </s>'
    # 目前写死了，top3
    contexts = example['passage'][:3]
    f = 'context:' + " {}"
    passages = [f.format(c['segment']) for c in contexts]

    return {
            'question' : question,
            'target' : target,
            'passages' : passages,
    }
def get_target(example):
    if 'target' in example:
        target = example['target']
        return target + ' </s>'
    elif 'answers' in example:
        return random.choice(example['answers']) + ' </s>'
    else:
        return None
def preprocess_function_nq(example, num_passage):
    question = "question:" + " " + example['question']
    target = get_target(example)
    if 'ctxs' in example and num_passage is not None:
        f = "title:" + " {} " + "context:" + " {}"
        if 'score' in example['ctxs'][0]:
            example['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)
        contexts = example['ctxs'][:num_passage]
        passages = [f.format(c['title'], c['text']) for c in contexts]
        if len(contexts) == 0:
            contexts = [question]
    else:
        passages = None

    return {
            'question' : question,
            'target' : target,
            'passages' : passages,
    }

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

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (target_ids, target_mask, passage_ids, passage_masks)