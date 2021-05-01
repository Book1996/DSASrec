import torch
import torch.nn as nn
import Constants as Constants
from Layers import SelfAttentionBlock
import torch.nn.functional as F


def get_mask_pad(seq):
    mask = seq.ne(Constants.PAD)
    mask = mask.to(seq.device)
    return mask


def get_mask_tri(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.tril(
        torch.ones((len_s, len_s), device=seq.device), 0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1).ne(Constants.PAD)  # b x ls x ls

    return subsequent_mask


def get_mask_attn(pad, tri):
    s2s = torch.bmm(pad.float().unsqueeze(2), pad.float().unsqueeze(1)).ne(Constants.PAD)
    mask_attn = s2s & tri
    mask_attn = mask_attn.to(tri.device)
    return mask_attn


class DSASRec(nn.Module):

    def __init__(self, max_len, items_count, emb_dim=50, ffn_dim=100, n_layers=2, dropout=[0.3, 0.3]):
        super(DSASRec, self).__init__()
        self.items_count = items_count
        self.emb_dim = emb_dim
        self.ffn_dim = ffn_dim
        self.max_len = max_len
        self.item_emb = nn.Embedding(items_count + 2, emb_dim, padding_idx=Constants.PAD)
        self.position_emb = nn.Embedding(max_len + 2, emb_dim, padding_idx=Constants.PAD)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.position = torch.arange(1, self.max_len + 1, device=self.device)
        self.pad = torch.tensor(0, device=self.device)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.layer_stack = nn.ModuleList([
            SelfAttentionBlock(self.emb_dim, self.ffn_dim, dropout=dropout[0])
            for _ in range(n_layers)
        ])
        self.position_emb.weight.data.uniform_(-0.5 / emb_dim, 0.5 / emb_dim)

        self.item_emb.weight.data.uniform_(-0.5 / emb_dim, 0.5 / emb_dim)
        self.emb_dropout = nn.Dropout(dropout[0])
        self.emb_dropout1 = nn.Dropout(dropout[1])

    def forward(self, seq):
        # mask
        mask_pad = get_mask_pad(seq)
        mask_tri = get_mask_tri(seq)
        mask_att = get_mask_attn(mask_pad, mask_tri)

        # build position
        masked_position = torch.where(mask_pad, self.position, self.pad)
        # build embedding
        posi_emb = self.position_emb(masked_position)
        sqe_emb = self.item_emb(seq)
        output = self.layer_norm(self.emb_dropout(sqe_emb + posi_emb))
        # self_attention
        for enc_layer in self.layer_stack:
            output, attn_weight = enc_layer(output, mask_pad=mask_pad, mask_attn=mask_att)
        return output, attn_weight

    def prediction(self, batch_sample):
        seq, pos, neg, target = batch_sample
        output, attn_weight = self.forward(seq)
        neg_emb = self.emb_dropout1(self.item_emb(neg))
        target_emb = self.emb_dropout1(self.item_emb(target).unsqueeze(1))

        last_output = output[:, -1, :].unsqueeze(1)  # b,1,d
        pre_item = torch.cat([target_emb, neg_emb], 1)  # b,s,d
        score = torch.bmm(pre_item, last_output.transpose(1, 2)).squeeze()  # b,s
        log_neg = self.pad
        log_pos = self.pad
        return log_pos, log_neg, score,attn_weight

    def loss(self, batch_sample):
        seq, pos, neg, target = batch_sample
        mask_pad = get_mask_pad(seq)
        output, attn_weight = self.forward(seq)
        neg = torch.where(mask_pad, neg, self.pad)
        neg_emb = self.emb_dropout1(self.item_emb(neg))
        pos_emb = self.emb_dropout1(self.item_emb(pos))
        sl = mask_pad.int().sum(1, keepdim=True).float()
        pos_score = torch.sum(pos_emb * output, 2)
        neg_score = torch.sum(neg_emb * output, 2)
        log_pos = -F.logsigmoid(pos_score + 1e-16) / sl
        log_neg = -F.logsigmoid(-neg_score + 1e-16) / sl
        log_pos = torch.where(mask_pad, log_pos, self.pad.float()).sum(1)
        log_neg = torch.where(mask_pad, log_neg, self.pad.float()).sum(1)
        score = self.pad
        return log_pos, log_neg, score
