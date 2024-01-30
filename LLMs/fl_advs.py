import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e12, beta = 1):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score1 = beta * (q @ k_t) / math.sqrt(d_tensor) # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score1)
        # print(score)

        # 4. multiply with Value
        v = score @ v

        return v, score, score1

class OneHeadAttentionAdv(nn.Module):

    def __init__(self, in_dim, d_model):
        super(OneHeadAttentionAdv, self).__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(in_dim, d_model, bias=False)
        self.w_k = nn.Linear(in_dim, d_model, bias=False)
        self.w_v = nn.Linear(in_dim, in_dim, bias=False)
        self.w_concat = nn.Linear(in_dim, in_dim, bias=False)
        
    def identity_QK(self):
        self.w_k.weight.data.copy_(torch.tensor(np.transpose(np.linalg.pinv(self.w_q.weight.detach()))))
        
    def filter_pattern(self, pattern):
        pattern_np = pattern.detach().numpy()
        w_q = self.w_q.weight.clone().detach().numpy()

        pattern_norm_sq = np.sum(pattern_np * pattern_np)
        for row in range(w_q.shape[0]):
            row_norm = np.sum(w_q[row] * w_q[row])
            proj_vec  = np.sum(pattern_np * w_q[row])/pattern_norm_sq * pattern_np
            w_q[row] = w_q[row] - proj_vec
        
        self.w_q.weight.data.copy_(torch.tensor(w_q))

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention,score1 = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        
        # 5. concat heads 
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // 1
        tensor = tensor.view(batch_size, length, 1, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
class LinearAggregator(nn.Module):
    def __init__(self, dim, bias = 0.0):
        super(LinearAggregator, self).__init__()
        
        self.dim = dim
        
        self.weight = torch.eye(dim*2)
        self.weight[0:dim,dim:] = -torch.eye(dim)
        self.weight[dim:,0:dim] = -torch.eye(dim)
        
        self.bias = - torch.ones(dim*2) * bias
        
        self.relu = torch.nn.ReLU()
    
    def forward(self, input: Tensor) -> Tensor:
        temp_tensor = torch.zeros((input.shape[0], self.dim*2))
        for l in range(input.shape[1]):
            temp_tensor = temp_tensor + self.relu(F.linear(input[:,l,:], self.weight, self.bias))
        return temp_tensor
    
class BiasLearner(nn.Module):
    def __init__(self, no, lx, dx, d_att, barrier = 0.1):
        super(BiasLearner, self).__init__()
        
        self.att_0 = OneHeadAttention_AMI(in_dim = dx, d_model = d_att)
        self.att_1 = OneHeadAttention_AMI(in_dim = dx, d_model = d_att)
        self.att_1.identity_QK()
        self.la = LinearAggregator(dx, bias = 0.0)
        
        self.no = no
        self.lx = lx
        self.dx = dx
        self.barrier = barrier
    
    def forward(self, x=None, random_pattern = None):
        if x == None:
            x = torch.rand(self.no,self.lx,self.dx)
        else:
            assert x.shape == (self.no,self.lx,self.dx), "x should have specified shape"
            
        if random_pattern == None:
            random_pattern = torch.rand(self.dx)
        else:
            assert random_pattern.shape == torch.Size([self.dx]), "pattern should have specified size"
            
        self.att_0.filter_pattern(random_pattern)
        self.att_1.w_v.weight.data.copy_(self.att_0.w_v.weight.data)
        self.att_1.w_concat.weight.data.copy_(self.att_0.w_concat.weight.data)
        
        z_filtered = self.att_0(x,x,x)
        z_identity = self.att_1(x,x,x)
        
        bias = torch.max(torch.abs(z_filtered - z_identity))
        bias = bias*(1+self.barrier)
        
        return bias