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
import time
import pandas as pd
import pickle

import os

from fl_advs import OneHeadAttention_AMI
from fl_advs import LinearAggregator

from transformers import AutoTokenizer, BertForPreTraining, BertModel, AutoModel
from datasets import load_dataset

parser = argparse.ArgumentParser(description='API attention Twitter experiments')
parser.add_argument('-e', '--eps', type=float, required=True)
parser.add_argument('-b', '--beta', type=float, default=0.01)
parser.add_argument('--D', type=int, default=10)
parser.add_argument('--times', type=int, default=100)
parser.add_argument('--runs', type=int, default=40)
parser.add_argument('-p', '--numproc', type=int, default=8)
parser.add_argument('--numdata', type=int, default=500)
# parser.add_argument('-r', '--numneurons', type=int, default=1000)
parser.add_argument('-m', '--mech', type=str, default='BitRand')
parser.add_argument('-o', '--output_path', type=str, default='./')
parser.add_argument('-s', '--seed', type=int, default=1)
args = parser.parse_args()

# for reproducibility
torch.manual_seed(args.seed)

NUM_PROCESS = args.numproc

def tpr_tnr(prediction, truth):
    confusion_vector = prediction / truth

    true_negatives = torch.sum(confusion_vector == 1).item()
    false_negatives = torch.sum(confusion_vector == float('inf')).item()
    true_positives = torch.sum(torch.isnan(confusion_vector)).item()
    false_positives = torch.sum(confusion_vector == 0).item()

    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), (true_positives + true_negatives) / (true_negatives + false_negatives + true_positives + false_positives) 

def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, NUM_PROCESS)]

    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)



def parallel_matrix_operation(func, arr):
    chunks = np.array_split(arr, NUM_PROCESS)
    
    
    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(func, chunks)
    
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


l = 20
m = 2
r = 768

def float_to_binary(x, m, n):
    x_abs = np.abs(x)
    x_scaled = round(x_abs * 2 ** n)
    res = '{:0{}b}'.format(x_scaled, m + n)
    if x >= 0:
        res = '0' + res
    else:
        res = '1' + res
    return res

# binary to float
def binary_to_float(bstr, m, n):
    sign = bstr[0]
#     print(int(sign))
    bs = bstr[1:]
    res = int(bs, 2) / 2 ** n
    if int(sign) == 49:
        res = -1 * res
    return res

def string_to_int(a):
    bit_str = "".join(x for x in a)
    return np.array(list(bit_str)).astype(int)


def join_string(a, num_bit=l, num_feat=r):
    res = np.empty(num_feat, dtype="S20")
    # res = []
    for i in range(num_feat):
        # res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
        res[i] = "".join(str(x) for x in a[i*l:(i+1)*l])
    return res


def float_bin(x):
    return float_to_binary(x, m, l-m-1)
    

def bin_float(x):
    return binary_to_float(x, m, l-m-1)


def OME(sample_feature_arr, eps=10.0, l=20, m=2):
    r = sample_feature_arr.shape[1]
    
    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = parallel_matrix_operation(float_to_binary_vec, sample_feature_arr)
    feat = parallel_apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

    rl = r * l
    alpha_ome = 100
    index_matrix_1 = np.array([alpha_ome / (1+ alpha_ome), 1/ (1+alpha_ome**3)]*int(l/2)) # np.array(range(l))
    index_matrix_0 = np.array([ (alpha_ome * np.exp(eps/rl)) /(1 + alpha_ome* np.exp(eps/rl))]*int(l) )
    p_1 = np.tile(index_matrix_1, (sample_feature_arr.shape[0], r))
    p_0 = np.tile(index_matrix_0, (sample_feature_arr.shape[0], r))

    p_temp = np.random.rand(p_0.shape[0], p_0.shape[1])
    perturb_0 = (p_temp > p_0).astype(int)
    perturb_1 = (p_temp > p_1).astype(int)

    perturb_feat = np.array(torch.where(torch.tensor(feat)>0, torch.tensor((perturb_1 + feat)%2), torch.tensor((perturb_0 + feat)%2)) )
    perturb_feat = parallel_apply_along_axis(join_string, axis=1, arr=perturb_feat)

    return torch.tensor(parallel_matrix_operation(binary_to_float_vec, perturb_feat), dtype=torch.float)


def OME_1(sample_feature_arr, eps=10.0, l=20, m=2):
    
    float_bin_2 = lambda x: float_to_binary(x, m, l-m-1)
    float_to_binary_vec_2 = np.vectorize(float_bin_2)
    bin_float_2 = lambda x: binary_to_float(x, m, l-m-1)
    binary_to_float_vec_2 = np.vectorize(bin_float_2)

    r = sample_feature_arr.shape[1]
    
    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = float_to_binary_vec_2(sample_feature_arr)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

    rl = r * l
    alpha_ome = 100
    index_matrix_1 = np.array([alpha_ome / (1+ alpha_ome), 1/ (1+alpha_ome**3)]*int(l/2)) # np.array(range(l))
    index_matrix_0 = np.array([ (alpha_ome * np.exp(eps/rl)) /(1 + alpha_ome* np.exp(eps/rl))]*int(l) )
    p_1 = np.tile(index_matrix_1, (sample_feature_arr.shape[0], r))
    p_0 = np.tile(index_matrix_0, (sample_feature_arr.shape[0], r))

    p_temp = np.random.rand(p_0.shape[0], p_0.shape[1])
    perturb_0 = (p_temp > p_0).astype(int)
    perturb_1 = (p_temp > p_1).astype(int)

    perturb_feat = np.array(torch.where(torch.tensor(feat)>0, torch.tensor((perturb_1 + feat)%2), torch.tensor((perturb_0 + feat)%2)) )
    perturb_feat = np.apply_along_axis(join_string, axis=1, arr=perturb_feat)

    perturb_feat = binary_to_float_vec_2(perturb_feat)
    return torch.squeeze(torch.tensor(perturb_feat, dtype=torch.float))#.cuda()


def BitRand(sample_feature_arr, eps=10.0, l=20, m=2):

    r = sample_feature_arr.shape[1]
    
    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = parallel_matrix_operation(float_to_binary_vec, sample_feature_arr)
    feat = parallel_apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

    rl = r * l
    sum_ = 0
    for k in range(l):
        sum_ += np.exp(2 * eps*k /l)
    alpha = np.sqrt((eps + rl) /( 2*r *sum_ ))
    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (sample_feature_arr.shape[0], r))
    p =  1/(1+alpha * np.exp(index_matrix*eps/l) )
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)

    perturb_feat = (perturb + feat)%2
    perturb_feat = parallel_apply_along_axis(join_string, axis=1, arr=perturb_feat)
    # print(perturb_feat)
    return torch.tensor(parallel_matrix_operation(binary_to_float_vec, perturb_feat), dtype=torch.float)



def BitRand_1(sample_feature_arr, eps, l=20, m=2, r=768):
    float_bin_2 = lambda x: float_to_binary(x, m, l-m-1)
    float_to_binary_vec_2 = np.vectorize(float_bin_2)
    bin_float_2 = lambda x: binary_to_float(x, m, l-m-1)
    binary_to_float_vec_2 = np.vectorize(bin_float_2)

    feat_tmp = float_to_binary_vec_2(sample_feature_arr)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat_tmp)
    sum_ = 0
    for k in range(l):
        sum_ += np.exp(2 * eps*k /l)
    alpha = np.sqrt((eps + r*l) /( 2*r *sum_ ))

    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (sample_feature_arr.shape[0], r))
    p =  1/(1+alpha * np.exp(index_matrix*eps/l) )
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    perturb_feat = (perturb + feat)%2
    perturb_feat = np.apply_along_axis(join_string, axis=1, arr=perturb_feat)
    perturb_feat = binary_to_float_vec_2(perturb_feat)
    return torch.squeeze(torch.tensor(perturb_feat, dtype=torch.float))#.cuda()


class Classifier(nn.Module):
    def __init__(self, n_inputs):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 2)

    def forward(self, x):
        x = self.fc1(x)
        probs = F.softmax(x, dim=1)
        return x, probs


print('Initializing data...')

MODEL = f"cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all"

def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", return_tensors='pt', truncation=True)

# train_ds, test_ds = load_dataset("cardiffnlp/tweet_topic_single", split=[f'train_coling2022[:{args.numdata}]', f'test_coling2022[:{args.numdata}]'])
train_ds, test_ds = load_dataset("cardiffnlp/tweet_topic_single", split=[f'train_coling2022', f'test_coling2022'])
tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=32)

encoded_train = train_ds.map(tokenize, batched=True)
encoded_test = test_ds.map(tokenize, batched=True)

encoded_train.set_format("pt", columns=["input_ids"], output_all_columns=True)
encoded_test.set_format("pt", columns=["input_ids"], output_all_columns=True)

print('Setting up patterns...')

pattern_id = 3268

# remove pattern from the dataset
train_input_ids = torch.where(encoded_train['input_ids'] == pattern_id, 0, encoded_train['input_ids'])
test_input_ids = torch.where(encoded_test['input_ids'] == pattern_id, 0, encoded_test['input_ids'])

print('Load embeddings...')

model = AutoModel.from_pretrained(MODEL, output_hidden_states = True)

train_data = model(train_input_ids)['hidden_states'][0]
test_data = model(test_input_ids)['hidden_states'][0]

sen1 = model(torch.tensor(tokenizer.encode("This film was probably inspired by Harry. Lorem ipsum dolor sit amet, consectetur adipiscing elit.")).unsqueeze(0))['hidden_states'][0]
pattern = sen1[0][7]

print(train_data.shape)



if args.mech == 'BitRand':
    mech = BitRand
    mech_1 = BitRand_1
elif args.mech == 'OME':
    mech = OME
    mech_1 = OME_1
else:
    print('Error mech')
    exit()


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

no = args.D
lx = train_data.shape[1]
dx = train_data.shape[2]

beta = args.beta
d_att = dx - 1

eps = args.eps
# runs = int((train_data.shape[0])/no)
runs = args.runs

num_thread = torch.get_num_threads()

adv = []

SAVE_NAME = f'{args.output_path}/Twitter_{args.eps}_{args.mech}_{args.beta}_{args.D}.pkl'

if os.path.isfile(SAVE_NAME):
    log = pd.read_pickle(SAVE_NAME)
else:
    log = pd.DataFrame() 

for j in range(runs):

    seed_process = args.seed * 1000 + j

    torch.manual_seed(j)

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
    
    print("Training attack...")

    def task_train(i):
        np.random.seed(seed_process*1000 + i)
        torch.manual_seed(seed_process*1000 + i)
        # Input
        x = train_data[np.random.randint(0, train_data.shape[0], no)]

        # Attack when no pattern
        x_0 = mech_1(x.reshape(no * lx, dx).detach().numpy(), eps=eps).reshape(no, lx, dx)

        z_filtered = head_pattern(x_0,x_0,x_0)
        z_identity = head_identity(x_0,x_0,x_0)

        z_0 = torch.cat((z_filtered, z_identity), dim = 2)
        z_0_out = la(z_0)

    #     indicator_0.append(torch.max(z_0_out).detach())
        ind_0 = torch.max(z_0_out).detach()

        # Attack when pattern

        x_1 = x.clone()
        x_1[0,0,:] = pattern

        x_1 = mech_1(x_1.reshape(no * lx, dx).detach().numpy(), eps=eps).reshape(no, lx, dx)

        z_filtered = head_pattern(x_1,x_1,x_1)
        z_identity = head_identity(x_1,x_1,x_1)

        z_1 = torch.cat((z_filtered, z_identity), dim = 2)
        z_1.reshape((z_1.shape[0], -1))
        z_1_out = la(z_1)

    #     indicator_1.append(torch.max(z_1_out).detach())
        ind_1 = torch.max(z_1_out).detach()

        return ind_0, ind_1


    torch.set_num_threads(1) # Required for multiprocessing 

    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        ret = list(tqdm(pool.imap(task_train, range(args.times), chunksize=1), total=args.times))

    torch.set_num_threads(num_thread)

    indicator_0 = [i[0] for i in ret]
    indicator_1 = [i[1] for i in ret]



    x_train = torch.cat((torch.tensor(indicator_0), torch.tensor(indicator_1)))
    y_train = torch.cat((torch.zeros(len(indicator_0)), torch.ones(len(indicator_0)))).long()
    x_train = x_train.reshape(len(indicator_0)*2,1)

    lr = 1e-1
    linear_model = Classifier(1)
    optimizer = optim.Adam(linear_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    # optimizer = optim.Adam(model.parameters(), lr=lr)
    from tqdm import tqdm

    for epoch in range(500):
        model.train()

        out, probs = linear_model(x_train)
        loss = criterion(out, y_train)

        loss.backward() 
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()         # a clean up step for PyTorch


    print(f'Success rate on training: {sum(linear_model(x_train)[1].max(1)[1] == y_train)/y_train.shape[0]}')

    print("Testing attack...")

    def task_test(i):
        np.random.seed(seed_process*1000 + i)
        torch.manual_seed(seed_process*1000 + i)
        # Input
        x = test_data[np.random.randint(0, test_data.shape[0], no)]

        # Attack when no pattern
        x_0 = mech_1(x.reshape(no * lx, dx).detach().numpy(), eps=eps).reshape(no, lx, dx)

        z_filtered = head_pattern(x_0,x_0,x_0)
        z_identity = head_identity(x_0,x_0,x_0)

        z_0 = torch.cat((z_filtered, z_identity), dim = 2)
        z_0_out = la(z_0)

    #     indicator_0.append(torch.max(z_0_out).detach())
        ind_0 = torch.max(z_0_out).detach()

        # Attack when pattern

        x_1 = x.clone()
        x_1[0,0,:] = pattern

        x_1 = mech_1(x_1.reshape(no * lx, dx).detach().numpy(), eps=eps).reshape(no, lx, dx)

        z_filtered = head_pattern(x_1,x_1,x_1)
        z_identity = head_identity(x_1,x_1,x_1)

        z_1 = torch.cat((z_filtered, z_identity), dim = 2)
        z_1.reshape((z_1.shape[0], -1))
        z_1_out = la(z_1)

    #     indicator_1.append(torch.max(z_1_out).detach())
        ind_1 = torch.max(z_1_out).detach()

        return ind_0, ind_1


    torch.set_num_threads(1) # Required for multiprocessing 

    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        ret = list(tqdm(pool.imap(task_test, range(args.times), chunksize=1), total=args.times))

    torch.set_num_threads(num_thread)

    indicator_0 = [i[0] for i in ret]
    indicator_1 = [i[1] for i in ret]

    x_test = torch.cat((torch.tensor(indicator_0), torch.tensor(indicator_1)))
    y_test = torch.cat((torch.zeros(len(indicator_0)), torch.ones(len(indicator_0)))).long()
    x_test = x_test.reshape(len(indicator_0)*2,1)
    
    predictions = linear_model(x_test)[1].max(1)[1]
    
    success_rate = sum(predictions == y_test)/y_test.shape[0]

    print(f'Success rate on testing: {success_rate}')
    adv.append(success_rate)
    
    result = {
                'run': j,
                'Success rate': success_rate,
                'wins': predictions & y_test,
                'game output': predictions,
                'y_test': y_test
             }

    log = pd.concat([log, pd.DataFrame.from_records([result])])
    with open(SAVE_NAME, 'wb') as logfile:
        pickle.dump(log, logfile)

        
print(np.mean(adv))