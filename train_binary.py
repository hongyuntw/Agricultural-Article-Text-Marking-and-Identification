import torch
from transformers import AdamW
from dataset import MyDataset, MyDataset_triples
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
epochs = 5
warm_up_rate = 0.03
json_path = './data/train_complete.json'
train_negative_nums = 10
multi_gpu = True
model_save_path = './outputs_models/binray_attention_from_colbert_weights/'
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
train_set = MyDataset(mode, json_data, tokenizer, train_negative_nums)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
warm_up_steps = total_steps * warm_up_rate
print(f'warm_up_steps : {warm_up_steps}')




colbert = ColBERT(pretrained_model, device=device)
colbert_model_path = './outputs_models/colbert/model_5.pt'
colbert.load_state_dict(torch.load(colbert_model_path), strict=False)


model = MyModel(pretrained_model)

model_params = model.named_parameters()
model_dict_params = dict(model_params)

colbert_params = colbert.named_parameters()
colbert_dict_params = dict(colbert_params)

for name, param in colbert_dict_params.items():
    if 'bert' in name:
        model_dict_params[name].data.copy_(param.data)
        model_dict_params[name].requires_grad = False
    
model.load_state_dict(model_dict_params, strict=False)

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_steps)


class_weight = torch.FloatTensor([1/2959, 1/1383]).to(device)
loss_fct = nn.CrossEntropyLoss(weight=class_weight)
# loss_fct = nn.CrossEntropyLoss()

model = model.to(device)
if multi_gpu:   
    model = nn.DataParallel(model)
model.train()


for epoch in range(epochs):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
    for i, data in enumerate(train_loader):        
        test_input_ids, test_attention_mask, test_token_type_ids, ref_input_ids, ref_attention_mask, ref_token_type_ids, labels = [t.to(device) for t in data]
        

        # forward pass
        outputs = model(test_input_ids, test_attention_mask, test_token_type_ids, ref_input_ids, ref_attention_mask, ref_token_type_ids)


        loss = loss_fct(outputs, labels)
        running_loss += loss.item()
        loss = loss / accumulation_steps

        loss.backward()

        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        acc += compute_acc(outputs, labels)

        print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , end='' )

    torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
    print(' saved ')