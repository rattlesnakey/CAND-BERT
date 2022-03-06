# -*- coding:UTF-8 -*-
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
# import wandb

def train_fn(data_loader, model, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        optimizer.zero_grad()
        #batch -> dict() -> input_ids, token_type_ids, attention_mask, labels
        batch_inputs = {k:v.to(device) for k, v in batch.items() if k != "labels"} 
        labels = batch['labels'].to(device)
        logits = model(batch_inputs)
        # print('train_logits:',logits.shape)
        # print('train_labels:',logits.shape)
        loss = F.cross_entropy(logits,labels)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# 最后还要写一个predict的函数
def eval_fn(data_loader, model, device, save_path, num_labels):
    print("________________________________________________")
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        total_trg = []
        total_hyp = []
        for batch in tqdm(data_loader, total=len(data_loader)):
            batch_inputs = {k:v.to(device) for k, v in batch.items() if k != "labels"} 
            labels = batch['labels'].to(device)
            logits = model(batch_inputs)  # batch_size, dim
            # print('eval_logits:',logits.shape)  # 这两个维度不一样
            # print('eval_labels:',labels.shape)
            loss = F.cross_entropy(logits,labels)  # batch_size, 1
            loss = loss.mean()
            epoch_loss += loss.item()
            batch_trg = labels.cpu().detach().numpy()
            batch_probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
            batch_hyp = np.argmax(batch_probs, 1)
            total_trg.append(batch_trg)
            total_hyp.append(batch_hyp)
            # wandb.log(dict(iter_loss=loss))
        total_hyp = np.concatenate(total_hyp)
        total_trg = np.concatenate(total_trg)
        # print(f"\n{total_hyp[:100]}")
        # print(total_trg[:100])
        epoch_avg_loss = epoch_loss / len(data_loader)
        val_acc, F1, recall, class_report, conf_matrix = evaluate(total_trg, total_hyp, num_labels)
    return val_acc, F1, recall, class_report, conf_matrix, epoch_avg_loss

def evaluate(targets, predicts, num_labels):
    target_names = [str(i) for i in range(num_labels)]
    val_acc = metrics.accuracy_score(targets, predicts)
    recall = metrics.recall_score(targets, predicts)
    F1 = metrics.f1_score(targets, predicts)
    class_report = metrics.classification_report(targets, predicts, target_names=target_names, digits=num_labels)
    conf_matrix = metrics.confusion_matrix(targets, predicts)
    return val_acc, F1, recall, class_report, conf_matrix
