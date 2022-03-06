from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from finetune_model import BertDefendantClassificationModel
from get_preprocessed_data import BERT_Preprocess_for_train
import torch
import numpy as np
from tqdm import tqdm
from train_and_eval import train_fn, eval_fn, evaluate
from data_utils import *
from param_config import Config
import warnings
import os 
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    


def main():
    warnings.filterwarnings("ignore")
    config = Config()

    # set random seed
    random_seed = config.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    train_data = BERT_Preprocess_for_train(config.train_data_path).output_data()
    dev_data = BERT_Preprocess_for_train(config.dev_data_path).output_data()

    logging.info("load data ...")
    if config.baseline:
        if not os.path.exists(config.baseline_prefix):
            os.makedirs(config.baseline_prefix)
        train_texts, train_labels = read_and_process_data(train_data, config.baseline)
        dev_texts, dev_labels = read_and_process_data(dev_data, config.baseline)
        logging.info(f'training_data:{len(train_texts)} samples')
        logging.info(f'dev_data:{len(dev_texts)} samples')
        logging.info("load tokenizer ...")
        tokenizer = BertTokenizer.from_pretrained(config.bert_path, tokenize_chinese_chars=True)  
        logging.info("encode input sents ...")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=config.max_len)
        dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, return_tensors="pt",max_length=config.max_len)

    else:
        if not os.path.exists(config.model1_prefix):
            os.makedirs(config.model1_prefix)
        train_texts, train_spans, train_labels = read_and_process_data(train_data, config.baseline)  
        dev_texts, dev_spans, dev_labels = read_and_process_data(dev_data, config.baseline)
        logging.info(f'training_data:{len(train_texts)} samples')
        logging.info(f'dev_data:{len(dev_texts)} samples')
        logging.info("load tokenizer ...")
        tokenizer = BertTokenizer.from_pretrained(config.bert_path, tokenize_chinese_chars=True) 
        tokenizer.add_special_tokens({"additional_special_tokens":["[CAND]"]})
        logging.info("encode input sents ...")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt", max_length=config.max_len)
        train_encodings = create_new_encodings(train_encodings, train_spans)
        dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, return_tensors="pt",max_length=config.max_len)
        dev_encodings = create_new_encodings(dev_encodings, dev_spans)

    logging.info("creating dataset ...")
    train_dataset = DefendantClassifyDataset(train_encodings, train_labels)
    dev_dataset = DefendantClassifyDataset(dev_encodings, dev_labels)
    
    logging.info("create dataloader ...")
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.eval_batch_size, shuffle=True)

    model = BertDefendantClassificationModel(config)
    model.bert.resize_token_embeddings(len(tokenizer))
    model = torch.nn.parallel.DataParallel(model)
    model.to(config.device)

    # optimizer settings
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
    num_training_steps = int(len(train_dataset) / (config.epochs * config.train_batch_size))

    # training
    best_acc = 0
    dev_accs_list = []
    logging.info("start training ...")
    for epoch in range(config.epochs):
        if config.baseline:
            save_path = config.baseline_prefix + f"Bertlaw_epoch{epoch+1}.pt"
        else:
            save_path = config.model1_prefix + f"Bertlaw_epoch{epoch+1}.pt"
        train_loss = train_fn(train_loader, model, optimizer, config.device)
        torch.save(model.module.state_dict(), save_path)
        dev_acc, dev_F1, dev_recall, class_report, conf_matrix, dev_loss = eval_fn(dev_loader, model, config.device, save_path, config.num_labels)
        dev_accs_list.append(dev_acc)
        if dev_acc > best_acc:
            best_acc = dev_acc
            es = 0
        else:
            es += 1
            logging.info(f"Counter {es} of {config.early_stopping}")
            if es > config.early_stopping:
                logging.info(f"Early stopping with best_acc: {best_acc}, and val_acc for this epoch:{dev_acc}")
                break
        logging.info(f"epoch {epoch+1} training finished!")
        logging.info(f"train_loss: {train_loss}")
        logging.info(f"dev_acc: {dev_acc}, dev_loss: {dev_loss}")
        logging.info(f"dev_F1: {dev_F1}, dev_loss: {dev_F1}")
        logging.info(f"dev_recall: {dev_recall}, dev_loss: {dev_recall}")
        logging.info(f"best performance on dev set appears in epoch {dev_accs_list.index(best_acc)+1}! ")
        print(class_report)



    

if __name__ == "__main__":  
    main()

