import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DeepAttention(nn.Module):
    def __init__(self, dim, temperature, dropout):
        super(DeepAttention, self).__init__()
        self.num = 2
        self.d = dim * self.num
        self.temperature = temperature
        self.deep = nn.Sequential(
            nn.Linear(dim * 2, 80),
            nn.ReLU(),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Linear(60, 1),
        )
        self.softmax = nn.Softmax(2)
        self.dropout = nn.Dropout(dropout)
        self.vw = nn.Linear(dim, dim)
        self.pad = torch.FloatTensor([-1e16]).cuda()
        self.pad0 = torch.FloatTensor([0.0]).cuda()

    def select(self, qk, index, T):
        qk = qk.reshape(-1, 2 * T)
        select_qk = qk[index, :]
        return select_qk, index

    def scatter(self, weight, index, B, S):
        att_zeros = torch.zeros((B * S * S), requires_grad=True, device=weight.device)
        att_weight = att_zeros.scatter(0, index, weight).reshape(B, S, S)
        return att_weight

    def forward(self, q, k, v, mask):
        B, S, T = q.shape
        # index
        index = torch.arange(B * S * S, device=q.device)
        index = torch.masked_select(index, mask.view(B * S * S))
        # deep part
        v = v
        q = q.unsqueeze(2).expand(-1, -1, S, -1)
        k = k.unsqueeze(1).expand(-1, S, -1, -1)
        qk = torch.cat([q, k], -1)
        qk, index = self.select(qk, index, T)
        attn = self.deep(qk).squeeze()
        attn = self.scatter(attn, index, B, S)
        attn = self.dropout(self.softmax(torch.where(mask, attn / self.temperature, self.pad)))
        output = torch.matmul(attn, v)  # b*s*t
        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, temperature, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(2)
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pad = torch.FloatTensor([-1e16]).cuda()

    def forward(self, q, k, v, mask):
        B, S, T = q.shape
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        attn = torch.matmul(q, k.transpose(1, 2))  # B,T,S,S
        attn = torch.where(mask, attn / self.temperature, self.pad)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # B,T,S,T
        return output, attn
