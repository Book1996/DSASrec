''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Modules import DeepAttention, ScaledDotProductAttention


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, dropout):
        super(SelfAttention, self).__init__()
        self.attention = ScaledDotProductAttention(emb_dim, np.power(emb_dim, 0.5), dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        residual = v
        output, attn = self.attention(q, k, v, mask)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, fnn_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(emb_dim, fnn_dim, 1)  # position-wise
        self.w_2 = nn.Conv1d(fnn_dim, emb_dim, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
