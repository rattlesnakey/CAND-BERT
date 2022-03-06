# -*- coding:UTF-8 -*-
import re
from transformers import BertTokenizer
import torch
import pickle
from tqdm import tqdm
class DefendantClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):  
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_new_encodings(encodings, spans):
    max_len = encodings['token_type_ids'].shape[1]
    new_token_type_ids = []
    for span in spans:
        span_start, span_end = span
        if span_start >= max_len:
            cur_token_type_id = [0 for _ in range(max_len)]
        if span_start < max_len and span_end <= max_len:
            cur_token_type_id = [0 for _ in range(span_start)] + [1 for _ in range(span_start, span_end + 1)] + [0 for _ in range(span_end + 1, max_len)]
        if span_start < max_len and span_end > max_len:
            cur_token_type_id = [0 for _ in range(span_start)] + [1 for _ in range(span_start, max_len)]
        new_token_type_ids.append(cur_token_type_id)
    encodings['token_type_ids'] = torch.tensor(new_token_type_ids)
    return encodings
       
def process_data(line, baseline):
    line = line.strip().split(' ||| ')
    cand, context, origin_factor, factor, label = line
    context_replaced_special = context.replace(cand, '@')
    origin_factor_replaced_special = origin_factor.replace(cand, '@')
    span_start = context_replaced_special.find(origin_factor_replaced_special)
    span_end = span_start + len(origin_factor_replaced_special)
    cand_token = "[CAND]"
    context_replaced_cand = context.replace(cand, cand_token)
    factor_replaced_cand = factor.replace(cand, cand_token)
    if baseline:
        return ("[SEP]".join([cand, context, factor]), label)
    return ("[SEP]".join([cand_token, context_replaced_cand, factor_replaced_cand]), span_start, span_end, label)


def read_and_process_data(data_list, baseline):
    texts, spans, labels = [], [], []
    if baseline:
        for line in data_list:
            text, label  = process_data(line, baseline)
            texts.append(text)
            labels.append(int(label))
        return texts, labels
    else:
        for line in data_list:
            text, span_start, span_end,label  = process_data(line, baseline)
            texts.append(text)
            spans.append((span_start, span_end))
            labels.append(int(label))
        return texts, spans, labels

def read_and_process_data_for_inference(data_list, evaluate):
    texts, spans, cands, origin_texts, origin_factors = [], [], [], [], []
    if evaluate:
        labels = []
        for line in (data_list):
            if line == '*':
                texts.append('*')
                spans.append('*')
                labels.append('*')
                cands.append('*')
                origin_texts.append('*')
                origin_factors.append('*')
            else:
                text, span_start, span_end, label, cand, context, origin_factor = process_data_for_inference(line, evaluate)
                texts.append(text)
                spans.append((span_start, span_end))
                labels.append(int(label))
                cands.append(cand)
                origin_texts.append(context)
                origin_factors.append(origin_factor)
        return texts, spans, labels, cands, origin_texts, origin_factors
    else:
        for line in (data_list):
            if line == '*':
                texts.append('*')
                spans.append('*')
                cands.append('*')
                origin_texts.append('*')
                origin_factors.append('*')
            else:
                text, span_start, span_end, cand, context, origin_factor= process_data_for_inference(line, evaluate)
                texts.append(text)
                spans.append((span_start, span_end))
                cands.append(cand)
                origin_texts.append(context)
                origin_factors.append(origin_factor)
        return texts, spans, cands, origin_texts, origin_factors




def process_data_for_inference(line, evaluate):
    line = line.strip().split(' ||| ')
    if evaluate:
        cand, context, origin_factor, factor, label = line
    else:
        cand, context, origin_factor, factor = line
    context_replaced_special = context.replace(cand, '@')
    origin_factor_replaced_special = origin_factor.replace(cand, '@')
    span_start = context_replaced_special.find(origin_factor_replaced_special)
    span_end = span_start + len(origin_factor_replaced_special)
    cand_token = "[CAND]"
    context_replaced_cand = context.replace(cand, cand_token)
    origin_factor_cand = origin_factor.replace(cand, cand_token)
    factor_replaced_cand = factor.replace(cand, cand_token)
    if evaluate:
        return ("[SEP]".join([cand_token, context_replaced_cand, factor_replaced_cand]), span_start, span_end, label, cand, context, origin_factor)
    else:
        return ("[SEP]".join([cand_token, context_replaced_cand, factor_replaced_cand]), span_start, span_end, cand, context, origin_factor)


