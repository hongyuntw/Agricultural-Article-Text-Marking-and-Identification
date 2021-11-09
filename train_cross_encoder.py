import torch
from transformers import AdamW, BertForSequenceClassification
from dataset import MyDataset, MyDataset_triples, BinaryDataset, CrossEncoderDataset
from model import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from preprocess import load_data_json
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from utils import compute_acc, compute_recall, compute_f1, compute_all, compute_precision
from torch.utils.data import random_split
from colbert import ColBERT
import os
from transformers import logging
logging.set_verbosity_warning()
### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'  
pretrained_model = 'hfl/chinese-macbert-base'  
lr = 1e-5
batch_size = 4
val_batch_size = 8
accumulation_steps = 1
mode = 'train'
epochs = 5
warm_up_rate = 0.03
json_path = './data/train_complete_ref_test_equal.json'
train_pairs_path = './data/train_binary_pairs'
train_hard_negative_nums = 20
train_rand_negative_nums = 20
multi_gpu = False
warm_up = False
valid = True
model_save_path = f'./outputs_models/cross_encoder_binary_hard{train_hard_negative_nums}_rand{train_rand_negative_nums}/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
### hyperparams ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)
if device == 'cpu':
    multi_gpu = False


json_data = load_data_json(json_path)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
train_set = CrossEncoderDataset(mode, json_data, tokenizer, train_hard_negative_nums, train_rand_negative_nums, train_pairs_path=None)
print(len(train_set))
# Random split
if valid:
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size
else:
    train_set_size = len(train_set)
    valid_set_size = 0
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(7777))
print(f'train_size : {train_set_size}, val_size {valid_set_size}')

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=val_batch_size, shuffle=False)

total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
warm_up_steps = total_steps * warm_up_rate
print(f'warm_up_steps : {warm_up_steps}')


model = BertForSequenceClassification.from_pretrained(pretrained_model)

optimizer = AdamW(model.parameters(), lr=lr)
if warm_up:
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_steps)


class_weight = torch.FloatTensor([1/16640, 1/2384]).to(device)
loss_fct = nn.CrossEntropyLoss(weight=class_weight)

model = model.to(device)
if multi_gpu:   
    model = nn.DataParallel(model)
model.train()


for epoch in range(epochs):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
    recall = 0.0
    f1 = 0.0
    precision = 0.0
    model.train()
    for i, data in enumerate(train_loader):        
        input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in data]

        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
        logits = outputs.logits


        loss = loss_fct(logits, labels)
        running_loss += loss.item()
        loss = loss / accumulation_steps

        loss.backward()
        if ((i+1) % accumulation_steps) or ((i+1) == len(train_loader)) == 0:
            optimizer.step()
            optimizer.zero_grad()
            if warm_up:
                scheduler.step()
        
        acc += compute_acc(logits, labels)
        precision += compute_precision(logits, labels)
        recall += compute_recall(logits, labels)
        f1 += compute_f1(logits, labels)
        print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}, precision {precision/ (i+1) :.5f} recall {recall/ (i+1) :.5f} f1 {f1/ (i+1) :.5f}' , end='' )
    
    print('')
    # valid 
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    totals_batch = len(valid_loader)
    val_recall = 0.0
    val_f1 = 0.0
    val_precision = 0.0
    for i, data in enumerate(valid_loader):        
        input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in data]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            logits = outputs.logits

            loss = loss_fct(logits, labels)
        
        val_loss += loss.item()
        val_acc += compute_acc(logits, labels)
        val_precision += compute_precision(logits, labels)
        val_recall += compute_recall(logits, labels)
        val_f1 += compute_f1(logits, labels)

        
        print(f'\r[val]Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {val_loss / (i+1) :.5f}, acc : {val_acc/ (i+1) :.5f}, precision {val_precision/ (i+1) :.5f} recall {val_recall/ (i+1) :.5f} f1 {val_f1/ (i+1) :.5f}' , end='' )

    torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
    print(' saved ')