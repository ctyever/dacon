import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelWithLMHead, GPT2ForSequenceClassification

warnings.filterwarnings('ignore')

tr = pd.read_csv('./data/train_data.csv', index_col='index')

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

model = GPT2ForSequenceClassification.from_pretrained("skt/kogpt2-base-v2")
model.score = torch.nn.Linear(768, 7)
model.cuda()

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=40):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        document, label = str(record['title']), int(record['topic_idx'])
        tokens = self.tokenizer.tokenize(document)
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoder_input_id)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [tokenizer.convert_tokens_to_ids('<pad>')]
                attention_mask += [0]
        else:
            encoder_input_id = encoder_input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(attention_mask, dtype=np.float),
                'labels': np.array(label, dtype=np.int_)}

class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=40):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        document = str(record['title'])
        tokens = self.tokenizer.tokenize(document)
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoder_input_id)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [tokenizer.convert_tokens_to_ids('<pad>')]
                attention_mask += [0]
        else:
            encoder_input_id = encoder_input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(attention_mask, dtype=np.float)}
    
# train parameters
epochs = 10
batch_size = 32

# train loader
train_ds = TrainDataset(tr, tokenizer)
loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, )
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for e in range(epochs):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        ids, atts, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        ids = torch.tensor(ids).long().cuda()
        atts = torch.tensor(atts).long().cuda()
        labels = torch.tensor(labels).long().cuda()
        pred = model(ids, attention_mask=atts)
        loss = loss_fn(pred[0], labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        total_loss += loss.item()
        
    scheduler.step()
    print(e, total_loss)

# test loader
te = pd.read_csv('./data/test_data.csv', index_col='index')

test_ds = TestDataset(te, tokenizer)
test_loader = DataLoader(test_ds, 8)

preds = []
model.eval()

for b in tqdm(test_loader):
    ids, atts = b['input_ids'], b['attention_mask']
    ids = torch.tensor(ids).long().cuda()
    atts = torch.tensor(atts).long().cuda()
    pred = model(ids, attention_mask=atts)
    preds += list(np.argmax(pred[0].detach().cpu().numpy(), 1))

sub = pd.read_csv('./data/sample_submission.csv', index_col='index')
sub['topic_idx'] = preds
sub.head(20)

sub.to_csv('./gpt.csv')