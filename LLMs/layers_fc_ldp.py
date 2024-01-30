import math
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import multiprocessing
import argparse
import os
import copy
import time
import pandas as pd
import pickle

from transformers import AutoTokenizer, BertForPreTraining, BertModel
from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import BertForPreTraining, BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
from transformers import RobertaModel, RobertaTokenizer
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from sklearn import  metrics

from ldpfunctions import ldp_mechanism

parser = argparse.ArgumentParser(description='API attention experiments')
parser.add_argument('--D', type=int, default=10)
parser.add_argument('--numdata', type=int, default=40)
parser.add_argument('-m', '--mode', type=str, default='full')
parser.add_argument('-s', '--seed', type=int, default=1)
parser.add_argument('-r', '--repeat', type=int, default=1)

args = parser.parse_args()

class Classifier(nn.Module):
    def __init__(self, n_inputs):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 2*n_inputs)
        self.fc2 = nn.Linear(2*n_inputs, 1)
        
    def adv_weights(self, target, tau, delta = 0):
        K = torch.eye(self.fc1.in_features)
        K = torch.cat((K, -K), 0)
        self.fc1.weight.data = K
        self.fc1.bias.data = torch.cat((-target - delta, target - delta), 0)
        
        self.fc2.weight.data[0] = -torch.ones(self.fc2.in_features)
        self.fc2.bias.data[0] = tau
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        fc2 = self.fc2(x)
        x = F.relu(fc2)
        return x, fc2


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, return_tensors='pt', truncation=True)

# List of dataset names and model names to test
dataset_names = ["imdb", "yelp", "twitter", "banking"]
model_names = ["bert", "roberta", "distilbert", "gpt2", "openai"]

for dataset_name in dataset_names:
    for model_name in model_names:
        print(f"Running dataset: {dataset_name}, model: {model_name}")

        if dataset_name == "yelp":
            train_ds, _ = load_dataset("yelp_polarity", split=[f'train[:{args.numdata}]', f'test[:{args.numdata}]'])
        elif dataset_name == "twitter":
            train_ds, _ = load_dataset("emotion", split=[f'train[:{args.numdata}]', f'test[:{args.numdata}]'])
        elif dataset_name == "banking":
            train_ds, _ = load_dataset("banking77", split=[f'train[:{args.numdata}]', f'test[:{args.numdata}]'])
        else:
            train_ds, _ = load_dataset("imdb", split=[f'train[:{args.numdata}]', f'test[:{args.numdata}]'])

        if model_name == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', model_max_length=32)
            model = RobertaModel.from_pretrained("roberta-base", output_hidden_states = True)
            offset = 1
            if dataset_name == "banking":
                size = 26
            else:
                size = 32
        elif model_name == "distilbert":
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', model_max_length=32)
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", output_hidden_states = True)
            offset = 1
            if dataset_name == "banking":
                size = 28
            else:
                size = 32
        elif model_name == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', model_max_length=32)
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2Model.from_pretrained('distilgpt2', output_hidden_states = True)
            offset = 0
            if dataset_name == "banking":
                size = 24
            else:
                size = 32
        elif model_name == "openai":
            tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt", model_max_length=32)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     tokenizer.pad_token = tokenizer.eos_token
            model = OpenAIGPTModel.from_pretrained("openai-gpt", output_hidden_states = True)
            model.resize_token_embeddings(len(tokenizer))
            offset = 0
            if dataset_name == "banking":
                size = 26
            else:
                size = 32
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=32)
            model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
            offset = 1
            if dataset_name == "banking":
                size = 28
            else:
                size = 32
        model.eval()

        num_hidden_layers = model.config.num_hidden_layers
        layers = [1, num_hidden_layers // 2, num_hidden_layers - 1]

        # Tokenizing
        encoded_train = train_ds.map(tokenize, batched=True)
        encoded_train.set_format("pt", columns=["input_ids"], output_all_columns=True)
            
        # Get pattern for testing
        print('Setting up clean patterns...')

        attack_sentence = "Find me if you can."
        for _ in range(10):
            attack_sentence = attack_sentence + " Find me if you can." 

        attack_ids = tokenizer.encode(attack_sentence)[0:size]

        input_ids_w0 = encoded_train['input_ids'].clone()
        input_ids_w1 = encoded_train['input_ids'].clone()
        
        del encoded_train
        
        input_ids_w1[0] = torch.Tensor(attack_ids)
        patterns = model(torch.tensor(attack_ids).unsqueeze(0))['hidden_states']
        mode = args.mode
        eps_s = [5, 7.5, 10, 0]
        
        for mechanism in ['GRR', 'RAPPOR', 'THE', 'dBitFlipPM']:
            for eps in eps_s:
                print("Mechanism: ", mechanism)
                for layer in layers:
                    print(f"Testing layer: {layer}")
                    sen_pattern = patterns[layer]
                    runs = 10
                    indicator_0 = []
                    indicator_1 = []
                    base_fpr = np.linspace(0, 1, 101)
                    tprs = []
                    aucs = []
                    f1s = []
                    accs = []
                    start_time = time.time()
                    for _ in range(runs):
                        if eps != 0:
                            input_ids_w1_ldp, _, _ = ldp_mechanism(input_ids_w1, eps, mechanism)
                            input_ids_w0_ldp, error2, error1 = ldp_mechanism(input_ids_w0, eps, mechanism)
                        else:
                            input_ids_w1_ldp = input_ids_w1.clone()
                            input_ids_w0_ldp = input_ids_w0.clone()
                            error2 = 0
                            error1 = 0

                        train_data_w0 = model(input_ids_w0_ldp)['hidden_states'][layer]
                        train_data_w1 = model(input_ids_w1_ldp)['hidden_states'][layer]

                        n, lx, dx = train_data_w0.shape

                        if mode == 'full':                    
                            train_data_w0_flat = train_data_w0.reshape((n, lx*dx))
                            train_data_w1_flat = train_data_w1.reshape((n, lx*dx))
                            fc_adv = Classifier(lx*dx)
                            fc_adv.adv_weights(sen_pattern.reshape(lx*dx), 0.0, 0.0)
                            fc_adv.eval()
                        else:
                            train_data_w0_flat = train_data_w0.reshape((n*lx, dx))
                            train_data_w1_flat = train_data_w1.reshape((n*lx, dx))
                            fc_adv = Classifier(dx)
                            fc_adv.adv_weights(sen_pattern.reshape(lx*dx)[dx*10:dx*11], 0.0, 0.0)
                            fc_adv.eval()

                        # Input
                        x_0 = train_data_w0_flat
                        x_1 = train_data_w1_flat

                        # Attack when no pattern
                        out_0 = fc_adv(x_0)[1]
                        indicator_0.append(torch.max(out_0).detach())

                        # Attack when pattern
                        out_1 = fc_adv(x_1)[1]
                        indicator_1.append(torch.max(out_1).detach())

                    y_true = np.zeros(2*len(indicator_1))
                    y_true[0:len(indicator_1)] = 1
                    y_pred = indicator_1 + indicator_0
                    y_pred = [value.detach().item() for value in y_pred]

                    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
                    auc = metrics.roc_auc_score(y_true, y_pred)

                    # Calculate the G-mean
                    gmean = np.sqrt(tpr * (1 - fpr))

                    if all(v == 1 for v in gmean):
                        index = round(len(thresholds) / 2)
                    else:
                        index = np.argmax(gmean)

                    mean_pred = sum(y_pred) / len(y_pred)
                    thresholdOpt = mean_pred

                    y_thres = [value > thresholdOpt for value in y_pred]
                    f1 = metrics.f1_score(y_true, y_thres)
                    acc = metrics.accuracy_score(y_true, y_thres)
                    tpr = np.interp(base_fpr, fpr, tpr)
                    tpr[0] = 0.0
                    tprs.append(tpr)
                    aucs.append(auc)
                    f1s.append(f1)
                    accs.append(acc)

                    tprs_logging = np.array(tprs)
                    mean_tprs = tprs_logging.mean(axis=0)
                    std = tprs_logging.std(axis=0)
                    fpr_logging = base_fpr
                    auc_logging = np.asarray(aucs)
                    f1_logging = np.asarray(f1s)
                    acc_logging = np.asarray(accs)
                    duration = time.time() - start_time

                    result_log = pd.DataFrame()
                    report_result = {
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Layer': layer,
                        'Accuracy': acc,
                        'F1': f1,
                        'AUC': auc,
                        'Eps': eps,
                        'Mechanism': mechanism,
                        'Duration': duration/runs,
                        'Error2': error2,
                        'Error1': error1,
                        'Mode': mode
                    }
                    print("TPRs: ", mean_tprs)
                    log_filename = f'result/fc_Oct01_' + mode + '_ldp.pkl'
                    if os.path.isfile(log_filename):
                        result_log = pd.read_pickle(log_filename)
                    else:
                        result_log = pd.DataFrame()

                    result_log = pd.concat([result_log, pd.DataFrame.from_records([report_result])])
                    with open(log_filename, 'wb') as logfile:
                        pickle.dump(result_log, logfile)
