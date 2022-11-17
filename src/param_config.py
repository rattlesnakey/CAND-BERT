import torch
class Config():
    def __init__(self):
        self.bert_path = "../bert-base-chinese"
        self.train_data_path = "../data/train.json"
        self.dev_data_path = "../data/dev.json"
        self.max_len = 512
        self.random_seed = 2
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_batch_size = 64
        self.eval_batch_size = 16
        self.epochs = 20 
        self.early_stopping = 3
        self.lr = 4e-5
        self.num_labels = 2
        self.dropout = 0
        self.requires_grad = True
        self.baseline = False
        self.baseline_prefix = "../saved_models/baseline/"
        self.model1_prefix = "../saved_models/CAND_BERT/"
        self.linear_hidden_size = 768

