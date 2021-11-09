import torch
from transformers import AdamW, BertForSequenceClassification
from dataset import MyDataset, TestDataset, CrossEncoderDataset
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
from tqdm import tqdm
from colbert import ColBERT
import pickle

### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'
pretrained_model = 'hfl/chinese-macbert-base'  

mode = 'test'
data_mode = 'train'
json_path = f'./data/{data_mode}_complete.json'
batch_size = 256
# model
binary_used_epoch = 2
binary_model_path = f'./outputs_models/cross_encoder_binary_neg60_alldata/model_{binary_used_epoch}.pt'

result_path = f'./output_results/{data_mode}_binary_result_cross_encoder_binary_neg60_alldata_epoch{binary_used_epoch}'
print(result_path)
### hyperparams ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)



json_data = load_data_json(json_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


test_set = CrossEncoderDataset(mode, json_data, tokenizer)
print(len(test_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load binary model cross encoder
binary_model = BertForSequenceClassification.from_pretrained(pretrained_model)
binary_model.load_state_dict(torch.load(binary_model_path), strict=False)
binary_model = binary_model.to(device)
binary_model = binary_model.eval()


# pred_list = []

results = {}
all_test_dids = []
all_ref_dids = []
all_binary_preds = []



with torch.no_grad():
    for i , data in enumerate(tqdm(test_loader)):
        input_ids, attention_mask, token_type_ids = [t.to(device) for t in data[0]]
        test_dids, ref_dids = data[1]
        all_test_dids += list(test_dids)
        all_ref_dids += list(ref_dids)


        binary_outputs = binary_model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        binary_outputs = binary_outputs.logits


        binary_preds = F.softmax(binary_outputs, dim=1).detach().cpu().numpy()[:, 1]
        all_binary_preds += list(binary_preds)

# all_binary_preds = [j for sub in all_binary_preds for j in sub]
# all_test_dids = [j for sub in all_test_dids for j in sub]
# all_ref_dids = [j for sub in all_ref_dids for j in sub]

 
for i in range(len(all_binary_preds)):
    test_did = all_test_dids[i]
    ref_did = all_ref_dids[i]
    binary_pred = all_binary_preds[i]
    if test_did in results:
        results[test_did].append((ref_did, binary_pred))
    else:
        results[test_did] = [(ref_did, binary_pred)]

with open(result_path, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(result_path)
