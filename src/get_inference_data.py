# -*- coding:UTF-8 -*-
import json
from tqdm import tqdm
from data_utils import *

class Preprocess_Data_for_Inference(object):
    def __init__(self, filename):
        self.f = json.load(open(filename,'r'))
        self.data = self.read_data()
    
    def read_data(self,):
        temp_dict = dict()
        for line in (self.f):
            for k, v in line.items():
                if k not in temp_dict.keys():
                    temp_dict[k] = []
                temp_dict[k].append(v)
        return temp_dict
    
    def filter_not_include_origin_factor_and_sentence(self, data):
        cur_filter_data, count = dict(), 0
        para_list, sentence_list, origin_factor_list = data['段落内容'], data['句子'], data['要素原始值'] 
        for idx, (para, sentence, origin_factor) in enumerate(zip(para_list, sentence_list, origin_factor_list)):
            if sentence in para and origin_factor in para:
                count += 1
                for k, v in data.items():
                    if k not in cur_filter_data.keys():
                        cur_filter_data[k] = []
                    cur_filter_data[k].append(v[idx])
        return cur_filter_data, count
    
    def filter_for_cands(self, data):
        cur_filter_data, count = dict(), 0
        cand_set_list, para_list = data['被告人集合'], data['段落内容'] 
        for idx, (cand_set, para) in (enumerate(zip(cand_set_list, para_list))):
            cand_set = list(set(cand_set))
            flag = 1
            for cand in cand_set:
                if cand not in para:
                    flag = 0
            if flag:
                count += 1
                for k, v in data.items():
                    if k not in cur_filter_data.keys():
                        cur_filter_data[k] = []
                    if k == '被告人集合':
                        cur_filter_data[k].append(cand_set)
                        continue 
                    cur_filter_data[k].append(v[idx])
        return cur_filter_data, count
    
    def preprocess(self, ):
        first_data, count_first = self.filter_not_include_origin_factor_and_sentence(self.data)
        processed_data, final_count = self.filter_for_cands(first_data)
        return processed_data, final_count       


class Extract_Context_Data(object):
    def __init__(self, data):
        self.data = data

    def ssplit(self, text):
        text = re.sub('([。；！？\?])([^”’])', r"\1\n\2", text)
        text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
        text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
        text = re.sub('([。；！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
        re.sub("\n+", "\n", text)
        return text.split("\n")

    def forward_sen(self, sens, idx, cur_cand):
        for sen_forward in sens[idx:]:
            if cur_cand in sen_forward:
                return sen_forward

    def backward_sen(self, sens, idx, cur_cand):
        for sen_backward in reversed(sens[:idx]):
            if cur_cand in sen_backward:
                return sen_backward
            
    def extract_context(self, para, cur_sen, cand):
        sens, cur_sens = self.ssplit(para), self.ssplit(cur_sen)
        for idx, sen in enumerate(sens):
            s = sen.strip()
            if cur_sens[0] in s:
                if idx == 0:
                    sen_forward = self.forward_sen(sens, idx, cand)
                    return sen_forward
                else:
                    sen_backward = self.backward_sen(sens, idx, cand)
                    if sen_backward == None:
                        return self.forward_sen(sens, idx, cand)
                    else:
                        return sen_backward
        
    def extract_context_for_all(self, ):
        new_sentence_list = []
        para_list, sentence_list, cand_set_list = self.data['段落内容'], self.data['句子'], self.data['被告人集合']
        for para, sentence, cand_set in (zip(para_list, sentence_list, cand_set_list)):
            cur_sentence = [sentence]
            lefts = cand_set.copy()
            for cand in cand_set.copy():
                if cand in sentence:
                    lefts.remove(cand)
            if len(lefts) == 0:
                new_sentence_list.append(''.join(cur_sentence))
                continue
            flag = 0
            for left in cand_set.copy():
                if left in lefts:
                    cur_return = self.extract_context(para, sentence, left)
                    cur_sentence.append(cur_return)
                    for cur_left in cand_set.copy():
                        if cur_left in lefts and cur_left in cur_sentence:
                            lefts.remove(cur_left)
                            if len(lefts) == 0:
                                flag = 1
                                break
                    if flag:
                        break
            new_sentence_list.append(''.join(cur_sentence))
        self.data['句子'] = new_sentence_list
        return self.data

class BERT_Preprocess_for_Inference(object):
    def __init__(self, indata_filename):
        self.init_data, _ = Preprocess_Data_for_Inference(indata_filename).preprocess()
        self.extracted_data = Extract_Context_Data(self.init_data).extract_context_for_all() 
        self.filter_data = self.furthur_filter_data()
        self.inference_data = []
    
    def furthur_filter_data(self, ):
        temp_dict = dict()
        for idx, (origin_factor, sentence) in enumerate(zip(self.extracted_data['要素原始值'], self.extracted_data['句子'])):
            if origin_factor in sentence:
                for k in self.extracted_data.keys():
                    if k not in temp_dict.keys():
                        temp_dict[k] = []
                    temp_dict[k].append(self.extracted_data[k][idx]) 
        return temp_dict
    
    def get_inference_data(self, evaluate = False):  # 有没有label
        temp_value_list = [v for k, v in self.filter_data.items()]
        for idx, t in (enumerate(zip(*temp_value_list))):
            # 如果输入的是带有label的
            if evaluate:
                para_id, para, cand_set, s, factor, origin_factor, defendants = t
                for cand in cand_set:
                    if cand in set(defendants):
                        temp_list = [cand, s, origin_factor, factor, str(1)]
                        cur_str = ' ||| '.join(temp_list)
                    else:
                        temp_list = [cand, s, origin_factor, factor, str(0)]
                        cur_str = ' ||| '.join(temp_list)
                    self.inference_data.append(cur_str)
            # 没有 label 的数据集
            else:
                para_id, para, cand_set, s, factor, origin_factor = t
                for cand in cand_set:
                    temp_list = [cand, s, origin_factor, factor]
                    cur_str = ' ||| '.join(temp_list)
                    self.inference_data.append(cur_str)
            # 分隔一个 sample
            self.inference_data.append('*')
        return self.inference_data  # ["cand ||| origin_factor ||| factor", ]




