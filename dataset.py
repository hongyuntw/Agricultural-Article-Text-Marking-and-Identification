import torch
from torch.utils.data import Dataset
import random
import ast


# for testing
class TestDataset(Dataset):
    def __init__(self, test_did, ref_dids, data, tokenizer):
        self.tok = tokenizer
        
        self.test_did = test_did
        self.did2text  = {}
        self.did2title = {}
        self.build_did2data(data)
        # list of id
        self.ref_dids = ref_dids

    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title

    def tensorsize(self, text):
            input_dict = self.tok(
                text,
                add_special_tokens=True,
                max_length=512,
                return_tensors='pt',
                pad_to_max_length=True,
                truncation='longest_first',
            )

            input_ids = input_dict['input_ids'][0]
            token_type_ids = input_dict['token_type_ids'][0]
            attention_mask = input_dict['attention_mask'][0]

            return (input_ids, attention_mask, token_type_ids)

    def __getitem__(self, idx):
        ref_did = self.ref_dids[idx]

        test_text = self.did2title[self.test_did] + self.did2text[self.test_did]
        ref_text = self.did2title[ref_did] + self.did2text[ref_did]

        test_input_ids , test_attention_mask, test_token_type_ids = self.tensorsize(test_text)
        ref_input_ids , ref_attention_mask, ref_token_type_ids = self.tensorsize(ref_text)

        return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask

            
    def __len__(self):
        return len(self.ref_dids)


        







# for train score model (ColBERT)
class MyDataset_triples(Dataset):
    def __init__(self, mode, data, tokenizer, negative_nums=2):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.negative_nums = negative_nums
        self.tok = tokenizer

        # (test_did, pos_ref_did, neg_ref_did)
        self.train_triples = []
        
        self.did2text  = {}
        self.did2title = {}
        self.all_dids = set()
        self.build_did2data(data)

        if mode == 'train':
            self.build_train_triples(data)
        if mode == 'test':
            self.build_test_pairs(data)

    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title
            self.all_dids.add(did)


    def build_train_triples(self,data):
        for d in data:
            did = d['did']
            pos_dids = d['pos_dids']
            neg_dids = d['hard_neg_dids']
            if len(pos_dids) == 0:
                continue

            # for pos_did in pos_dids:
            #     for neg_did in neg_dids[:self.negative_nums]:
            #         self.train_triples.append((did, pos_did, neg_did))

            pos_dids_set = set(pos_dids)
            for pos_did in pos_dids:
                for neg_did in self.all_dids:
                    if neg_did in pos_dids_set or neg_did == did:
                        continue
                    self.train_triples.append((did, pos_did, neg_did))
                    




    def build_test_pairs(self, data):
        pass
        # all_dids = sorted(list(self.did2text.keys()))
        # for test_did in all_dids:
        #     for ref_did in all_dids:
        #         if ref_did != test_did:
        #             self.test_pairs.append((test_did, ref_did))

    def tensorsize(self, text):

        input_dict = self.tok(
            text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        return (input_ids, attention_mask, token_type_ids)

    def combine_ref(self, pos_ref, neg_ref):
        pos_ref_input_ids, pos_ref_attention_mask, pos_ref_token_type_ids = pos_ref
        neg_ref_input_ids, neg_ref_attention_mask, neg_ref_token_type_ids = neg_ref


        input_ids = torch.cat((pos_ref_input_ids.unsqueeze(0), neg_ref_input_ids.unsqueeze(0)))
        attention_mask = torch.cat((pos_ref_attention_mask.unsqueeze(0), neg_ref_attention_mask.unsqueeze(0)))
        token_type_ids = torch.cat((pos_ref_token_type_ids.unsqueeze(0), neg_ref_token_type_ids.unsqueeze(0)))
 
        return (input_ids, attention_mask, token_type_ids)

    def __getitem__(self, idx):
        if self.mode == 'train':
            test_did , pos_ref_did, neg_ref_did = self.train_triples[idx]
        # if self.mode == 'test':
        #     test_did , ref_did = self.test_pairs[idx]

        test_text = self.did2title[test_did] + self.did2text[test_did]
        pos_ref_text = self.did2title[pos_ref_did] + self.did2text[pos_ref_did]
        neg_ref_text = self.did2title[neg_ref_did] + self.did2text[neg_ref_did]

        test = self.tensorsize(test_text)
        pos_ref = self.tensorsize(pos_ref_text)
        neg_ref = self.tensorsize(neg_ref_text)

        ref = self.combine_ref(pos_ref, neg_ref)

        test_input_ids , test_attention_mask, test_token_type_ids = test
        ref_input_ids , ref_attention_mask, ref_token_type_ids = ref
        if self.mode == 'train':
            # return test_input_ids , test_attention_mask, test_token_type_ids, ref_input_ids , ref_attention_mask, ref_token_type_ids
            return test_input_ids , test_attention_mask, ref_input_ids , ref_attention_mask

            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_triples)
        # if self.mode == 'test':
        #     return len(self.test_pairs)

class BinaryDataset(Dataset):
    def __init__(self, mode, data, tokenizer,  train_pairs_path):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.did2text  = {}
        self.did2title = {}
        self.all_dids = set()
        self.tok = tokenizer

        self.build_did2data(data)
        import pickle
        with open(train_pairs_path, 'rb') as handle:
            self.train_pairs =  pickle.load(handle)


    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title
            self.all_dids.add(did)

    def tensorsize(self, text):

        input_dict = self.tok(
            text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        return (input_ids, attention_mask, token_type_ids)
         
    def __getitem__(self, idx):
        if self.mode == 'train':
            test_did , ref_did, label = self.train_pairs[idx]
            label = torch.tensor(label)

        if self.mode == 'test':
            test_did , ref_did = self.test_pairs[idx]


        test_text = self.did2title[test_did] + self.did2text[test_did]
        ref_text = self.did2title[ref_did] + self.did2text[ref_did]

        test_input_ids, test_attention_mask, test_token_type_ids = self.tensorsize(test_text)

        ref_input_ids, ref_attention_mask, ref_token_type_ids = self.tensorsize(ref_text)
        
        if self.mode == 'train':
            return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, label
        if self.mode == 'test':
            return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, test_did, ref_did 




    def __len__(self):
        if self.mode == 'train':
            return len(self.train_pairs)
        




# for train binary
class MyDataset(Dataset):
    def __init__(self, mode, data, tokenizer, negative_nums=2):
        assert mode in ["train", 'val',  "test"]
        self.mode = mode
        self.negative_nums = negative_nums
        self.tok = tokenizer
        # list of list , elemets be like (test_did, reference_did, label)
        self.train_pairs = []
        
        self.test_pairs = []
        self.did2text  = {}
        self.did2title = {}
        self.all_dids = set()
        self.build_did2data(data)

        if mode == 'train':
            self.build_train_pairs(data)
        if mode == 'test':
            self.build_test_pairs(data)

    def build_did2data(self, data):
        for d in data:
            use_text = ''.join(d['replaced_text_sentence'].split())
            use_title = ''.join(d['replaced_title_sentence'].split())
            did = d['did']
            self.did2text[did] = use_text
            self.did2title[did] = use_title
            self.all_dids.add(did)


    def build_train_pairs(self, data):
        pos_count = 0
        neg_count = 0
        for d in data:
            did = d['did']
            pos_dids = d['pos_dids']
            neg_dids = d['hard_neg_dids']

            pos_dids = set(pos_dids)

            if len(pos_dids) == 0:
                continue
            for pos_did in pos_dids:
                self.train_pairs.append((did, pos_did, 1))
                pos_count += 1
            
            for neg_did in neg_dids[:self.negative_nums]:
                self.train_pairs.append((did, neg_did, 0))
                neg_count += 1
            # for neg_did in self.all_dids:
            #     if neg_did != did and neg_did not in pos_dids:
            #         self.train_pairs.append((did, neg_did, 0))
            #         neg_count += 1
        print(f'training set postive rate == {pos_count/(pos_count + neg_count) :.3f}, pos={pos_count} neg={neg_count}')

    def build_test_pairs(self, data):
        all_dids = sorted([int(k) for k in self.did2text.keys()])
        for test_did in all_dids:
            for ref_did in all_dids:
                if ref_did != test_did:
                    self.test_pairs.append((str(test_did), str(ref_did)))

    def tensorsize(self, text):

        input_dict = self.tok(
            text,
            add_special_tokens=True,
            max_length=512,
            return_tensors='pt',
            pad_to_max_length=True,
            truncation='longest_first',
        )

        input_ids = input_dict['input_ids'][0]
        token_type_ids = input_dict['token_type_ids'][0]
        attention_mask = input_dict['attention_mask'][0]

        return (input_ids, attention_mask, token_type_ids)
         
    def __getitem__(self, idx):
        if self.mode == 'train':
            test_did , ref_did, label = self.train_pairs[idx]
            label = torch.tensor(label)

        if self.mode == 'test':
            test_did , ref_did = self.test_pairs[idx]


        test_text = self.did2title[test_did] + self.did2text[test_did]
        ref_text = self.did2title[ref_did] + self.did2text[ref_did]

        test_input_ids, test_attention_mask, test_token_type_ids = self.tensorsize(test_text)

        ref_input_ids, ref_attention_mask, ref_token_type_ids = self.tensorsize(ref_text)
        
        if self.mode == 'train':
            return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, label
        if self.mode == 'test':
            return test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, test_did, ref_did 


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_pairs)
        if self.mode == 'test':
            return len(self.test_pairs)