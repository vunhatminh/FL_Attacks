from sklearn.metrics import accuracy_score, recall_score
import numpy as np
from sys import maxsize
import xxhash
import copy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from numba import jit
import torch
# multi-freq-ldpy import
from multi_freq_ldpy.pure_frequency_oracles.UE import UE_Client
from multi_freq_ldpy.pure_frequency_oracles.LH import LH_Client
from multi_freq_ldpy.pure_frequency_oracles.SS import SS_Client
from multi_freq_ldpy.pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator_MI, GRR_Aggregator_IBU
from multi_freq_ldpy.pure_frequency_oracles.HE import HE_Client, HE_Aggregator_MI, HE_Aggregator_IBU
from multi_freq_ldpy.long_freq_est.dBitFlipPM import dBitFlipPM_Client, dBitFlipPM_Aggregator_MI
from multi_freq_ldpy.long_freq_est.L_SUE import L_SUE_Client, L_SUE_Aggregator_MI, L_SUE_Aggregator_IBU

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import optimize

def find_tresh(tresh, epsilon):    
    
    return (2 * (np.exp(epsilon*tresh/2)) - 1) / (1 + (np.exp(epsilon*(tresh-1/2))) - 2*(np.exp(epsilon*tresh/2)))**2

def LH_Client_Fast(input_data, k, epsilon, optimal=True):
    
    # Binary LH (BLH) parameter
    g = 2
    
    # Optimal LH (OLH) parameter
    if optimal:
        g = int(round(np.exp(epsilon))) + 1
    
    # GRR parameters with reduced domain size g
    p = np.exp(epsilon) / (np.exp(epsilon) + g - 1)
    q = 1 / (np.exp(epsilon) + g - 1)
    
    # Generate random seed and hash the user's value
    rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
    hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)
    
    # LH perturbation function (i.e., GRR-based)
    sanitized_value = hashed_input_data
    rnd = np.random.random()
    if rnd > p - q:
        
        sanitized_value = np.random.randint(0, g)
        
    return (sanitized_value, rnd_seed)


def get_preprocessed_encoded_sets_with_ldp(df, target, test_size, seed, lst_sensitive_att, epsilon, split_strategy, lst_k, mechanism='GRR'):
    def LH_Client_high_eps(input_data, k, epsilon, optimal=True):
        """Backup function for OLH mechanism.
        Due to high epsilon values, the new
        domain size g is excessively high.
        Multi-freq-ldpy fails with np.random.randint.
        This new function uses np.random.uniform."""

        # Binary LH (BLH) parameter
        g = 2

        # Optimal LH (OLH) parameter
        if optimal:
            g = int(round(np.exp(epsilon))) + 1

        # Generate random seed and hash the user's value
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)

        # LH perturbation function (i.e., GRR-based)
        p = np.exp(epsilon) / (np.exp(epsilon) + g - 1)
        if np.random.random() <= p:
            sanitized_value = int(np.random.uniform() * g)
            while sanitized_value == hashed_input_data:
                sanitized_value = int(np.random.uniform() * g)        
            return (sanitized_value, rnd_seed)

        return (hashed_input_data, rnd_seed)
    
    # Use original dataset
    X = copy.deepcopy(df.drop(target, axis=1))
    y = copy.deepcopy(df[target])

    # Train test splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=seed)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    # One-Hot-Encoding + LDP randomization
    lst_df_train = []
    lst_df_test = []
    for col in list(set(df.columns) - set([target])):

        lst_col_name = [col+"_{}".format(val) for val in range(len(set(df[col])))]
        k = len(set(df[col]))
        OHE = np.eye(k)

        if col in lst_sensitive_att: # LDP randomization
            eps_att = epsilon / len(lst_sensitive_att) if split_strategy=='uniform' else epsilon * k / sum(lst_k.values())
            
            if mechanism == 'GRR':
                df_ohe = pd.DataFrame([OHE[GRR_Client(val, k, eps_att)] for val in X_train[col]], columns=lst_col_name)
                
            elif mechanism == 'BLH':
                df_ohe = pd.DataFrame([IVE_LH(LH_Client(val, k, eps_att, optimal=False), k, eps_att, optimal=False) for val in X_train[col]], columns=lst_col_name)
            
            elif mechanism == 'OLH':
                if eps_att <= 21:
                    df_ohe = pd.DataFrame([IVE_LH(LH_Client_Fast(val, k, eps_att, optimal=True), k, eps_att, optimal=True) for val in X_train[col]], columns=lst_col_name)
                else:
                    df_ohe = pd.DataFrame([IVE_LH(LH_Client_high_eps(val, k, eps_att, optimal=True), k, eps_att, optimal=True) for val in X_train[col]], columns=lst_col_name)
            
            elif mechanism == 'SUE':
                df_ohe = pd.DataFrame(np.stack(X_train[col].apply(lambda x: UE_Client(x, k, eps_att, optimal=False))), columns=lst_col_name) 
                
            elif mechanism == 'OUE':
                df_ohe = pd.DataFrame(np.stack(X_train[col].apply(lambda x: UE_Client(x, k, eps_att, optimal=True))), columns=lst_col_name)
                
            elif mechanism == 'SS':
                df_ohe = pd.DataFrame([IVE_SS(SS_Client(val, k, eps_att), k) for val in X_train[col]], columns=lst_col_name)
                
            elif mechanism == 'THE':
                res_tresh = optimize.minimize_scalar(find_tresh, bounds=[0,1], method='bounded', args=(eps_att))
                tresh_att = res_tresh.x

                df_ohe = pd.DataFrame([IVE_THE(HE_Client(val, k, eps_att), k, tresh_att) for val in X_train[col]], columns=lst_col_name)
            
            else:
                raise ValueError("Mechanism unknown!")
        else: # just one-hot-encoding
            df_ohe = pd.DataFrame([OHE[val] for val in X_train[col]], columns=lst_col_name)

        lst_df_train.append(df_ohe)

        # test set is original, i.e., just one-hot-encoding
        df_ohe_test = pd.DataFrame([OHE[val] for val in X_test[col]], columns=lst_col_name)
        lst_df_test.append(df_ohe_test)

    # concat one-hot-encoded train/test sets
    X_train = pd.concat(lst_df_train, axis=1)
    X_test = pd.concat(lst_df_test, axis=1)
    
    return X_train, X_test, y_train, y_test

   
def fairness_metrics(df_fm, protected_attribute, target):
    
    fair_met = {# Statistical Parity
                "SP_a_1": np.nan,
                "SP_a_0": np.nan,
                "DI": np.nan, # Disparate Impact
                "SPD": np.nan, # Statistical Parity Difference
                # Equal Opportunity
                "EO_a_1": np.nan,
                "EO_a_0": np.nan,
                "EOD": np.nan, # Equal Opportunity Difference
                # Overall Accuracy
                "OA_a_1": np.nan,
                "OA_a_0": np.nan,
                "OAD": np.nan, # Overall Accuracy Difference
                }
    
    # Filtering datasets for fairness metrics
    df_a_1 = df_fm.loc[df_fm[protected_attribute+"_1"]==1]
    df_a_0 = df_fm.loc[df_fm[protected_attribute+"_1"]==0]

    # Calculate Statistical Parity per group
    SP_a_1 = df_a_1.loc[df_a_1["y_pred"]==1].shape[0] / df_a_1.shape[0]
    SP_a_0 = df_a_0.loc[df_a_0["y_pred"]==1].shape[0] / df_a_0.shape[0]
    fair_met["SP_a_1"] = SP_a_1
    fair_met["SP_a_0"] = SP_a_0

    # Disparate Impact
    DI = SP_a_0 / SP_a_1
    fair_met["DI"] = DI
    
    # Statistical Parity Difference
    SPD = SP_a_1 - SP_a_0
    fair_met["SPD"] = SPD

    # Equal Opportunity
    EO_a_1 = recall_score(df_a_1[target], df_a_1['y_pred'])
    EO_a_0 = recall_score(df_a_0[target], df_a_0['y_pred'])
    fair_met["EO_a_1"] = EO_a_1
    fair_met["EO_a_0"] = EO_a_0

    # Equal Opportunity Difference
    EOD = EO_a_1 - EO_a_0
    fair_met["EOD"] = EOD

    # Overall Accuracy
    OA_a_1 = accuracy_score(df_a_1[target], df_a_1['y_pred'])
    OA_a_0 = accuracy_score(df_a_0[target], df_a_0['y_pred'])
    fair_met["OA_a_1"] = OA_a_1
    fair_met["OA_a_0"] = OA_a_0

    # Accuracy per Group Difference
    OAD = OA_a_1 - OA_a_0
    fair_met["OAD"] = OAD
    
    return fair_met    



def IVE_LH(val_seed, k, epsilon, optimal=True):
    """
    Indicator-Vector-Encoding (IVE) for Local Hashing (LH) Mechanisms.
    """
    
    g=2 # BLH parameter
    if optimal:
        g = int(np.round(np.exp(epsilon))) + 1 # OLH parameter
    
    ive_lh = np.zeros(k)

    for v in range(k):
        if val_seed[0] == (xxhash.xxh32(str(v), seed=val_seed[1]).intdigest() % g):
            ive_lh[v] = 1
    
    return ive_lh    
    
def IVE_SS(ss, k):
    """
    Indicator-Vector-Encoding (IVE) for Subset Selection (SS) Mechanism.
    """
    
    ive_ss = np.zeros(k)
    ive_ss[ss] = 1
    
    return ive_ss    
    
def IVE_THE(hist, k, thresh):
    """
    Indicator-Vector-Encoding (IVE) for Thresholding with Histogram Encoding (THE) Mechanism.
    """
    
    ss_the = np.where(hist > thresh)[0]
    
    ive_the = np.zeros(k)
    
    if len(ss_the) > 0:
        ive_the[ss_the] = 1
    
    return ive_the

def tensor_to_np(tensor):
    n,d = tensor.shape
    
    tensor_flat = tensor.reshape(n*d)
    keys = torch.unique(tensor_flat).detach()
    key_list = []
    for key in keys:
        key_list.append(key.item())
    tensor_to_list = dict(zip(key_list, range(keys.size()[0])))
    list_to_tensor_dict = dict(zip(range(keys.size()[0]), key_list))
    
    list_out = []
    for v in range(len(tensor_flat)):
        list_out.append(tensor_to_list[tensor_flat[v].item()])
    return np.asarray(list_out), list_to_tensor_dict

def ldp_mechanism(tensor, epsilon = 10, mechanism = 'GRR'):
    n,d = tensor.shape
    
    array1d, list_to_tensor_dict = tensor_to_np(tensor)
    real_freq = np.unique(array1d, return_counts=True)[-1]/n
    
    no_tokens = int(np.max(array1d)+1)
    OHE = np.eye(no_tokens)
    if mechanism == 'GRR': # Generalized Randomized Response
        out = [GRR_Client(entry, no_tokens, epsilon) for entry in array1d]
        est = GRR_Aggregator_MI(out, no_tokens, epsilon)
#         est = GRR_Aggregator_IBU(out, no_tokens, epsilon)
        ohe = [np.argmax(OHE[entry]) for entry in out]
        error2 = mean_squared_error(real_freq, est)
        error1 = mean_absolute_error(real_freq, est)
    elif mechanism == 'dBitFlipPM': # Microsoft
        out = [dBitFlipPM_Client(entry, no_tokens, no_tokens, no_tokens, epsilon) for entry in array1d]
        est = dBitFlipPM_Aggregator_MI(out, no_tokens, no_tokens, epsilon)
        ohe = [np.argmax(entry) for entry in out]
        error2 = mean_squared_error(real_freq, est)
        error1 = mean_absolute_error(real_freq, est)
    elif mechanism == 'RAPPOR': # Google
        out = [L_SUE_Client(entry, no_tokens, epsilon*2, epsilon) for entry in array1d]
        est = L_SUE_Aggregator_MI(out, epsilon*2, epsilon)
        ohe = [np.argmax(entry) for entry in out]
        error2 = mean_squared_error(real_freq, est)
        error1 = mean_absolute_error(real_freq, est)
    elif mechanism == 'THE': # Historgram threshold
        out = [HE_Client(entry, no_tokens, epsilon) for entry in array1d]
        est = HE_Aggregator_MI(out, no_tokens, epsilon, use_thresh=False)
        ohe = [np.argmax(entry) for entry in out]
        error2 = mean_squared_error(real_freq, est)
        error1 = mean_absolute_error(real_freq, est)
    else:
        raise ValueError("Mechanism unknown!")
#     print(array1d)
#     print(ohe)
    
    tensor_list = [list_to_tensor_dict[entry] for entry in ohe]
    tensor_ldp = torch.Tensor(tensor_list)
    return tensor_ldp.reshape((n,d)).type(torch.LongTensor), error2, error1