import torch
from transformers import AdamW
from dataset import MyDataset, TestDataset
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
json_path = './data/train_complete.json'
batch_size = 24
# model
binary_used_epoch = 3
binary_model_path = f'./outputs_models/binray_attention_from_colbert_weights_freeze/model_{binary_used_epoch}.pt'

score_used_epoch = 2
score_model_path = f'./outputs_models/colbert_neg30/model_{score_used_epoch}.pt'

### hyperparams ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)



json_data = load_data_json(json_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


test_set = MyDataset(mode, json_data, tokenizer)
print(len(test_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load binary model
# binary_model = MyModel(pretrained_model)
# binary_model.load_state_dict(torch.load(binary_model_path), strict=False)
# binary_model = binary_model.to(device)
# binary_model = binary_model.eval()


# load score model
score_model = ColBERT(pretrained_model, device=device)
score_model.load_state_dict(torch.load(score_model_path), strict=False)
score_model = score_model.to(device)
score_model = score_model.eval()



# pred_list = []

results = {}
all_test_dids = []
all_ref_dids = []
all_score_preds = []

with torch.no_grad():
    for i , data in enumerate(tqdm(test_loader)):
        test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask = [t.to(device) for t in data[:-2]]
        test_dids, ref_dids = data[-2:]

        D_test = (test_input_ids, test_attention_mask)
        D_ref = (ref_input_ids, ref_attention_mask)

        # binary_outputs = binary_model(D_test, D_ref)
        score_outputs = score_model(D_test, D_ref)

        # binary_preds = F.softmax(binary_outputs, dim=1).detach().cpu().numpy()

        all_test_dids += list(test_dids)
        all_ref_dids += list(ref_dids)

        score_preds = score_outputs.detach().cpu().numpy()
        # print(score_preds.shape)
        # score_preds = score_preds.ravel()
        # print(score_preds.shape)
        # input()

        all_score_preds += list(score_preds)
 

        # for i in range(score_preds.shape[0]):
        #     test_did = test_dids[i]
        #     ref_did = ref_dids[i]
        #     # binary_pred = binary_preds[i][1]
        #     binary_pred = 0
        #     score_pred = score_preds[i]
        #     if test_did in results:
        #         results[test_did].append((ref_did, binary_pred, score_pred))
        #     else:
        #         results[test_did] = [(ref_did, binary_pred, score_pred)]


for i in range(len(all_score_preds)):
    test_did = all_test_dids[i]
    ref_did = all_ref_dids[i]
    score_pred = all_score_preds[i]
    binary_pred = 0
    if test_did in results:
        results[test_did].append((ref_did, binary_pred, score_pred))
    else:
        results[test_did] = [(ref_did, binary_pred, score_pred)]

with open('./output_results/train_result_colbert30', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            









# with torch.no_grad():
#     for i , d in enumerate(tqdm(json_data)):
#         test_did = d['did']
#         ref_dids = [d_['did'] for d_ in json_data if d_['did'] != test_did]
#         test_set = TestDataset(test_did, ref_dids, json_data, tokenizer)
#         test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#         binary_preds = []
#         score_preds = []
#         for data in test_loader:
#             test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask = [t.to(device) for t in data]
        
#             # forward pass
#             D_test = (test_input_ids, test_attention_mask)
#             D_ref = (ref_input_ids, ref_attention_mask)
                
#             binary_outputs = binary_model(D_test, D_ref)
#             score_outputs = score_model(D_test, D_ref)

#             binary_pred = F.softmax(binary_outputs, dim=1).detach().cpu().numpy()
#             binary_preds.append(binary_pred)

#             score_pred = score_outputs.detach().cpu().numpy()
#             score_preds.append(score_pred)
        
#         binary_preds = np.vstack(binary_preds)
#         score_preds = np.hstack(score_preds)
#         print(binary_preds.shape)
#         print(score_preds.shape)





