import torch
from transformers import AdamW
from dataset import MyDataset
from model import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from preprocess import load_data_json
from transformers import AutoTokenizer
from utils import compute_acc
import os
import numpy as np
import torch.nn.functional as F

### hyperparams ###
pretrained_model = 'hfl/chinese-bert-wwm'
mode = 'test'
json_path = './data/train_complete.json'
multi_gpu = True
batch_size = 4
# for load model
used_epoch = 5
model_path = f'./outputs_models/single_encoder_concat/model_{used_epoch}.pt'
### hyperparams ###


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print("device:", device)
if device == 'cpu':
    multi_gpu = False


json_data = load_data_json(json_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
test_set = MyDataset(mode, json_data, tokenizer)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


model = MyModel(pretrained_model)
model.load_state_dict(torch.load(model_path), strict=False)
model = model.to(device)
if multi_gpu:   
    model = nn.DataParallel(model)
model = model.eval()

pred_list = []
with torch.no_grad():
    for i, data in enumerate(test_loader):   
        test_input_ids, test_token_type_ids, test_attention_mask, ref_input_ids, ref_token_type_ids, ref_attention_mask = [t.to(device) for t in data[:-2]]
        test_did, ref_did = data[-2:]
        outputs = model(test_input_ids, test_token_type_ids, test_attention_mask, ref_input_ids, ref_token_type_ids, ref_attention_mask)
        
        pred = F.softmax(outputs, dim=1).detach().cpu().numpy()
        pred = np.argmax(pred, axis=-1)

        # print((pred == 1))
        
        # test_did = np.array(test_did)
        # ref_did = np.array(ref_did)
        # pos_pairs = test_did[pred == 1] , ref_did
        # print(test_did)
        # print(ref_did)

        # print(pred.shape)
        # print(pred)
        # input()

