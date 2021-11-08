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
from utils import compute_acc
from colbert import ColBERT
import os
from transformers import logging
logging.set_verbosity_warning()
### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'  
pretrained_model = 'hfl/chinese-macbert-base'  
lr = 1e-5
batch_size = 4
accumulation_steps = 1
mode = 'train'
epochs = 10
warm_up_rate = 0.03
json_path = './data/train_complete.json'
train_pairs_path = './data/train_binary_pairs'
train_negative_nums = 50
multi_gpu = False
warm_up = False
model_save_path = './outputs_models/cross_encoder_binary_use_train_pairs/'
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
train_set = CrossEncoderDataset(mode, json_data, tokenizer, train_negative_nums, train_pairs_path=train_pairs_path)
print(len(train_set))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
warm_up_steps = total_steps * warm_up_rate
print(f'warm_up_steps : {warm_up_steps}')


model = BertForSequenceClassification.from_pretrained(pretrained_model)

optimizer = AdamW(model.parameters(), lr=lr)
if warm_up:
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_steps)


class_weight = torch.FloatTensor([1/13721, 1/1383]).to(device)
loss_fct = nn.CrossEntropyLoss(weight=class_weight)

model = model.to(device)
if multi_gpu:   
    model = nn.DataParallel(model)
model.train()


for epoch in range(epochs):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
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

        print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , end='' )

    torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
    print(' saved ')