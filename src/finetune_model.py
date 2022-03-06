from transformers import BertModel
import torch.nn as nn

class BertDefendantClassificationModel(nn.Module):

    def __init__(self, config):
        super(BertDefendantClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.config = config
        for param in self.bert.parameters():
            param.requires_grad = config.requires_grad  # 为什么这里关掉了?，因为数据量太少了，如果微调的话，可能就把它调坏了
        # self.dropout = nn.Dropout(p=config.dropout)
        self.linear = nn.Linear(config.linear_hidden_size, config.num_labels)

    def forward(self, data):
        all_last_hidden, pooled = self.bert(**data, output_hidden_states=False)
        if self.config.baseline:
            out = self.linear(pooled)
        else:
            out = self.linear(all_last_hidden[:,1,:]) # 把[CAND] token取出来
        # drop_pooled = self.dropout(pooled)
        # 这个地方如果不Squeze就不会报错, 因为output出来其实就是batch_size, dim的维度了，并没有时间
        # 然后如果squeeze的话，因为Paralllel可能分到batch_size为1的，这时候就会被约成一个维度，就错了
        # out = out.squeeze()
        return out