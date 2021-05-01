''' Define the Layers '''
import torch.nn as nn
from SubLayers import SelfAttention, PositionwiseFeedForward


class SelfAttentionBlock(nn.Module):
    def __init__(self, emb_dim, fnn_dim, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.self_attention = SelfAttention(emb_dim, dropout=dropout)
        self.ffn = PositionwiseFeedForward(emb_dim, fnn_dim, dropout=dropout)

    def forward(self, seq_emb, mask_pad, mask_attn):
        output, attn_weight = self.self_attention(seq_emb, seq_emb, seq_emb, mask_attn)
        output = self.ffn(output)
        output *= mask_pad.unsqueeze(2).float()
        return output, attn_weight
