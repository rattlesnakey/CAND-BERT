# -*- coding:UTF-8 -*-
import torch
from finetune_model import BertDefendantClassificationModel
from param_config import Config
from get_inference_data import BERT_Preprocess_for_Inference  
from data_utils import * 
import numpy as np
import json
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from pywebio import config

class Init(object):
    def __init__(self, best_model_path, evaluate = False):
        self.config = Config()
        self.tokenizer = self.load_tokenizer()
        self.best_model = self.load_best_model(best_model_path)
    
    def load_tokenizer(self,):
        tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, tokenize_chinese_chars=True)
        tokenizer.add_special_tokens({"additional_special_tokens":["[CAND]"]})
        return tokenizer

    def load_best_model(self, best_model_path):
        model = BertDefendantClassificationModel(self.config)
        model.bert.resize_token_embeddings(len(self.tokenizer))
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

class Predictor(object):
    def __init__(self, indata_filename, init, evaluate = False):
        self.config = Config()
        self.initial_inference_data = BERT_Preprocess_for_Inference(indata_filename).get_inference_data(evaluate)  # 竖线隔开的, 有*号
        self.best_model = init.best_model
        self.tokenizer = init.tokenizer
        self.texts, self.spans, self.labels, self.cands, self.origin_texts, self.origin_factors = None, None, None, None, None, None
        self.get_data(evaluate)

    def get_data(self, evaluate):
        re = read_and_process_data_for_inference(self.initial_inference_data, evaluate)
        if evaluate:
            self.texts, self.spans, self.labels, self.cands, self.origin_texts, self.origin_factors = re 
        else:
            self.texts, self.spans, self.cands, self.origin_texts, self.origin_factors = re

    def evaluate(self, ):
        pred_labels, true_labels = [], []
        acc_count, total = 0, 0
        for text, span, label in zip(self.texts, self.spans, self.labels):
            if text == '*':
                total += 1
                if pred_labels == true_labels:
                    acc_count += 1
                pred_labels, true_labels = [], []
                continue
            text_encoding = self.tokenizer([text], truncation=True, padding=True, return_tensors="pt", max_length=self.config.max_len)
            new_text_encoding = create_new_encodings(text_encoding, [span])
            pred = self.best_model(new_text_encoding)
            pred_probs = torch.softmax(pred, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(pred_probs, 1)
            pred_labels.append(pred_label)
            true_labels.append(label)
        print('全部正确的准确率:{}'.format(acc_count / total))

    def predict(self, ):
        pred_results, count = [], 0
        for idx, (text, span, cand) in enumerate(zip(self.texts, self.spans, self.cands)):
            if text == '*':
                last_origin_text, last_origin_factor= self.origin_texts[idx-1], self.origin_factors[idx-1]
                print('*'*100)
                print("案件上下文:",last_origin_text)
                print()
                print("案件要素原始值:", last_origin_factor)
                print()
                print("当前案件对应该案件要素原始值预测的被告人:",pred_results)
                print('*'*100)
                return {"案件上下文":last_origin_text, "案件要素原始值":last_origin_factor, "当前案件对应该案件要素原始值预测的被告人":pred_results}
                pred_results = []
                continue
            text_encoding = self.tokenizer([text], truncation=True, padding=True, return_tensors="pt", max_length=self.config.max_len)
            new_text_encoding = create_new_encodings(text_encoding, [span])
            pred = self.best_model(new_text_encoding)  
            pred_probs = torch.softmax(pred, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(pred_probs, 1)
            if pred_label == 1:
                pred_results.append(cand)


css = """
body {
    background-image: url(https://s2.loli.net/2021/12/12/3knXp28BKxhdriU.png);
    background-size: 100% 125%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    
}
body > footer {
    visibility:hidden;
}
"""


@config(css_style=css)
def main_web():
    # best_model_path = '../saved_models/CAND_BERT/model1Bertlaw_epoch18.pt'
    # init = Init(best_model_path, evaluate = False)
    data = input_group("法律要素关联性分析在线系统",[
        input('请输入文号:', name='content_id'),
        input('请输入段落内容:', name='para_content'),
        input('请输入被告人集合(逗号隔开):', name='cand_set'),
        input('请输入句子内容:', name='sen_content'),
        input('请输入要素名称:', name='factor'),
        input('请输入要素原始值:', name='origin_factor')
    ])
    temp_dict = [{'文号':data["content_id"], '段落内容':data["para_content"], '被告人集合':data["cand_set"].strip().split(','), \
    '句子':data["sen_content"], '要素名称':data["factor"], '要素原始值':data["origin_factor"]}, ]
    # put_loading
    json.dump(temp_dict, open('../data/temp_test.json','w+'))
    predictor = Predictor('../data/temp_test.json', init, evaluate = False)
    put_markdown('# 输出结果')
    with put_loading():
    # time.sleep(3)  # Some time-consuming operations
    # put_text("The answer of the universe is 42")
        popup('结果分析中...')
        pred_result = predictor.predict()
        close_popup()
        put_table([
            ['结果', '结果内容'],
            ['案件上下文', pred_result['案件上下文']],
            ['案件要素原始值', pred_result["案件要素原始值"]],
            ["当前案件对应该案件要素原始值预测的被告人", ",".join(pred_result["当前案件对应该案件要素原始值预测的被告人"])]
        ])

if __name__ == '__main__':
    best_model_path = '../saved_models/CAND_BERT/model1Bertlaw_epoch18.pt'
    init = Init(best_model_path, evaluate = False)
    start_server(main_web, port=8023)
    # while True:
    #     # content_id = input('请输入文号:')
    #     # print()
    #     # para_content = input('请输入段落内容:')
    #     # print()
    #     # cand_set = input('请输入被告人集合(逗号隔开):')
    #     # print()
    #     # cand_set = cand_set.strip().split(',')
    #     # sen_content = input('请输入句子内容:')
    #     # print()
    #     # factor = input('请输入要素名称:')
    #     # print()
    #     # origin_factor = input('请输入要素原始值:')
    #     # print()
    #     content_id, para_content  = "（2014）沙刑初字第169号", "经审理查明，被告人万某某于2012年9月至2013年11月10日间，在未取得客运许可的情况下，通过允许自带车辆加入的方式，提供印发服务卡片、制定服务规则、确定收费标准、设立电话总台、保证往返次数等支持，组建了由多辆无营运资格的私家车组成的往返于大连市长兴岛至大连市内四区拼客的固定线路营运车队“宏驰骋快捷车队”，并定期收取司机费用获利，违法所得人民币42000余元。 被告人王某某、陈某某分别于2012年9月至2013年11月10日期间，自带车辆加入由万某某组建的、无营运资格的私家车组成的、往返于大连市长兴岛至大连市内四区拼客的固定线路营运车队“宏驰骋快捷车队”，从事非法经营活动。被告人王某某违法所得人民币32000余元；被告人陈某某违法所得人民币12000余元。"
    #     cand_set, sen_content = ["万某某","王某某","陈某某"], "被告人王某某、陈某某分别于2012年9月至2013年11月10日期间，自带车辆加入由万某某组建的、无营运资格的私家车组成的、往返于大连市长兴岛至大连市内四区拼客的固定线路营运车队“宏驰骋快捷车队”，从事非法经营活动"
    #     factor, origin_factor = "其他严重扰乱市场秩序的非法经营行为", "自带车辆加入由万某某组建的、无营运资格的私家车组成的、往返于大连市长兴岛至大连市内四区拼客的固定线路营运车队“宏驰骋快捷车队”，从事非法经营活动"
    #     temp_dict = [{'文号':content_id, '段落内容':para_content, '被告人集合':cand_set, '句子':sen_content, '要素名称':factor, '要素原始值':origin_factor}, ]
    #     json.dump(temp_dict, open('../data/temp_test.json','w+'))
    #     predictor = Predictor('../data/temp_test.json', init, evaluate = False)
    #     predictor.predict()
    #     break
