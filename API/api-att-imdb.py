import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import multiprocessing
import argparse

import os

from fl_advs import OneHeadAttention_AMI
from fl_advs import LinearAggregator

from transformers import AutoTokenizer, BertForPreTraining, BertModel
from datasets import load_dataset

parser = argparse.ArgumentParser(description='API attention IMDB experiments')
parser.add_argument('-b', '--beta', type=float, default=0.1)
parser.add_argument('--D', type=int, default=10)
parser.add_argument('--numdata', type=int, default=1000)
parser.add_argument('-m', '--mech', type=str, default='BitRand')
parser.add_argument('-s', '--seed', type=int, default=1)
args = parser.parse_args()

# for reproducibility
torch.manual_seed(args.seed)

class Classifier(nn.Module):
    def __init__(self, n_inputs):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 2)

    def forward(self, x):
        x = self.fc1(x)
        probs = F.softmax(x, dim=1)
        return x, probs

print('Initializing data...')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, return_tensors='pt', truncation=True)

train_ds, test_ds = load_dataset("imdb", split=[f'train[:{args.numdata}]', f'test[:{args.numdata}]'])
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", model_max_length=32)

encoded_train = train_ds.map(tokenize, batched=True)
encoded_test = test_ds.map(tokenize, batched=True)

encoded_train.set_format("pt", columns=["input_ids"], output_all_columns=True)
encoded_test.set_format("pt", columns=["input_ids"], output_all_columns=True)

print('Setting up patterns...')

pattern_id = 3466

# remove pattern from the dataset
train_input_ids = torch.where(encoded_train['input_ids'] == pattern_id, 0, encoded_train['input_ids'])
test_input_ids = torch.where(encoded_test['input_ids'] == pattern_id, 0, encoded_test['input_ids'])

print('Load embeddings...')

model = BertModel.from_pretrained("bert-base-cased", output_hidden_states = True)

train_data = model(encoded_train['input_ids'])['hidden_states'][0]
test_data = model(encoded_test['input_ids'])['hidden_states'][0]

print(train_data.shape)

print('Setting up patterns...')

sen1 = model(torch.tensor(tokenizer.encode("This film was probably inspired by Harry. Lorem ipsum dolor sit amet, consectetur adipiscing elit.")).unsqueeze(0))['hidden_states'][0]
pattern = sen1[0][7]

no = args.D
lx = train_data.shape[1]
dx = train_data.shape[2]

beta = args.beta
d_att = dx - 1

runs = int((train_data.shape[0])/no)

success_rate = []

for seed in range(10):
    
    print("Training attack...")

    torch.manual_seed(seed)
    # Layer
    head_pattern = OneHeadAttention_AMI(in_dim = dx, d_model = d_att)
    head_pattern.filter_pattern(pattern)
    head_pattern.identity_QK()
    head_pattern.w_q.weight.data = head_pattern.w_q.weight.data*beta

    head_identity = OneHeadAttention_AMI(in_dim = dx, d_model = d_att)
    head_identity.identity_QK()
    head_identity.w_q.weight.data = head_identity.w_q.weight.data*beta
    head_identity.w_v.weight.data.copy_(head_pattern.w_v.weight.data)
    head_identity.w_concat.weight.data.copy_(head_pattern.w_concat.weight.data)

    la = LinearAggregator(dx, bias = 0.0)

    from tqdm import tqdm
    indicator_0 = []
    indicator_1 = []

    for i in tqdm(range(runs)):

        # Input
        x_0 = train_data[i*no:i*no+no]

        # Attack when no pattern

        z_filtered = head_pattern(x_0,x_0,x_0)
        z_identity = head_identity(x_0,x_0,x_0)

        z_0 = torch.cat((z_filtered, z_identity), dim = 2)
        z_0_out = la(z_0)

        indicator_0.append(torch.max(z_0_out).detach())

        # Attack when pattern

        x_1 = x_0.clone()
        x_1[0,0,:] = pattern
        z_filtered = head_pattern(x_1,x_1,x_1)
        z_identity = head_identity(x_1,x_1,x_1)

        z_1 = torch.cat((z_filtered, z_identity), dim = 2)
        z_1.reshape((z_1.shape[0], -1))
        z_1_out = la(z_1)

        indicator_1.append(torch.max(z_1_out).detach())

        if torch.max(z_0_out).detach() == torch.max(z_1_out).detach():
            indicator_0.pop()
            indicator_1.pop()


    x_train = torch.cat((torch.tensor(indicator_0), torch.tensor(indicator_1)))
    y_train = torch.cat((torch.zeros(len(indicator_0)), torch.ones(len(indicator_0)))).long()
    x_train = x_train.reshape(len(indicator_0)*2,1)

    lr = 1e-2
    linear_model = Classifier(1)
    optimizer = optim.Adam(linear_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    from tqdm import tqdm

    for epoch in range(500):
        model.train()

        out, probs = linear_model(x_train)
        loss = criterion(out, y_train)

        loss.backward() 
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()         # a clean up step for PyTorch


    print("Testing attack...")

    indicator_0 = []
    indicator_1 = []
    for i in tqdm(range(runs)):

        # Input
        x_0 = test_data[i*no:i*no+no]

        # Attack when no pattern

        z_filtered = head_pattern(x_0,x_0,x_0)
        z_identity = head_identity(x_0,x_0,x_0)

        z_0 = torch.cat((z_filtered, z_identity), dim = 2)
        z_0_out = la(z_0)

        indicator_0.append(torch.max(z_0_out).detach())

        # Attack when pattern

        x_1 = x_0.clone()
        x_1[0,0,:] = pattern
        z_filtered = head_pattern(x_1,x_1,x_1)
        z_identity = head_identity(x_1,x_1,x_1)

        z_1 = torch.cat((z_filtered, z_identity), dim = 2)
        z_1.reshape((z_1.shape[0], -1))
        z_1_out = la(z_1)

        indicator_1.append(torch.max(z_1_out).detach())

    x_test = torch.cat((torch.tensor(indicator_0), torch.tensor(indicator_1)))
    y_test = torch.cat((torch.zeros(len(indicator_0)), torch.ones(len(indicator_0)))).long()
    x_test = x_test.reshape(len(indicator_0)*2,1)

    success_rate.append(sum(linear_model(x_test)[1].max(1)[1] == y_test)/y_test.shape[0])
    
print(f'Success rate on testing: {np.mean(success_rate)}')
