import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer

max_len = 80


class ISLRModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.1,
        n_layer=1,
        max_len=80,
        n_labels=250,
        num_points=82,
        point_dim=3,
        batch_first=True,
    ):
        super(ISLRModel, self).__init__()
        pos_encoding = positional_encoding(max_len, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))
        self.xyz_emb = nn.Sequential(
            nn.Linear(num_points * point_dim, embed_dim, bias=False)
        )
        self.feature_extractor = TransformerBlock(
            embed_dim=embed_dim,
            n_head=n_head,
            ff_dim=ff_dim,
            dropout=dropout,
            n_layer=n_layer,
            batch_first=batch_first,
        )
        self.logit = nn.Linear(embed_dim, n_labels)

        self.max_len = max_len
        self.point_dim = point_dim

    def forward(self, batch):
        xyz = batch["xyz"]
        x, x_mask = pack_seq(xyz, self.max_len, self.point_dim)
        B, L, _ = x.shape
        x = self.xyz_emb(x)
        x = x + self.pos_emb[:L].unsqueeze(0)
        x = torch.cat([self.cls_emb.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)
        x = self.feature_extractor(x, x_mask)
        cls = x[:, 0]
        cls = F.dropout(cls, p=0.4, training=self.training)
        logit = self.logit(cls)

        return logit


class ISLRModelV2(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.1,
        n_layer=1,
        max_len=384,
        n_labels=250,
        input_dim=912,
        batch_first=True,
    ):
        super(ISLRModelV2, self).__init__()
        pos_encoding = positional_encoding(max_len, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))
        self.xyz_emb = nn.Sequential(nn.Linear(input_dim, embed_dim, bias=False))
        self.feature_extractor = TransformerBlock(
            embed_dim=embed_dim,
            n_head=n_head,
            ff_dim=ff_dim,
            dropout=dropout,
            n_layer=n_layer,
            batch_first=batch_first,
        )
        self.logit = nn.Linear(embed_dim, n_labels)

        self.max_len = max_len
        self.input_dim = input_dim

    def forward(self, batch):
        xyz = batch["xyz"]
        x, x_mask = pack_seqv2(xyz, self.max_len)
        B, L, _ = x.shape
        x = self.xyz_emb(x)
        x = x + self.pos_emb[:L].unsqueeze(0)
        x = torch.cat([self.cls_emb.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)
        x = self.feature_extractor(x, x_mask)
        cls = x[:, 0]
        cls = F.dropout(cls, p=0.4, training=self.training)
        logit = self.logit(cls)

        return logit


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.1,
        n_layer=1,
        batch_first=True,
    ):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(
            embed_dim, n_head, dropout=dropout, batch_first=batch_first
        )
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(ff_dim)
        # encoder_layer = TransformerEncoderLayer(
        #     d_model=embed_dim,
        #     nhead=n_head,
        #     dim_feedforward=ff_dim,
        #     dropout=dropout,
        #     batch_first=batch_first,
        # )
        # self.encoder = TransformerEncoder(encoder_layer, n_layer)

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        dropout,
        batch_first,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=n_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, x, x_mask):
        out, _ = self.mha(x, x, x, key_padding_mask=x_mask)
        return out


def positional_encoding(length, embed_dim):
    dim = embed_dim // 2
    position = np.arange(length)[:, np.newaxis]  # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :] / dim  # (1, dim)
    angle = 1 / (10000**dim)  # (1, dim)
    angle = position * angle  # (pos, dim)
    pos_embed = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


def pack_seq(seq, max_len, point_dim):
    length = [min(len(s), max_len) for s in seq]
    batch_size = len(seq)
    K = seq[0].shape[1]
    L = max(length)
    x = torch.zeros((batch_size, L, K, point_dim)).to(seq[0].device)
    x_mask = torch.zeros((batch_size, L)).to(seq[0].device)
    for b in range(batch_size):
        l = length[b]
        x[b, :l] = seq[b][:l]
        x_mask[b, l:] = 1
    x_mask = x_mask > 0.5
    x = x.reshape(batch_size, -1, K * point_dim)
    return x, x_mask


def pack_seqv2(seq, max_len):
    # seq: [L, 912]
    length = [min(len(s), max_len) for s in seq]
    batch_size = len(seq)
    K = seq[0].shape[-1]
    L = max(length)
    x = torch.zeros((batch_size, L, K)).to(seq[0].device)
    x_mask = torch.zeros((batch_size, L)).to(seq[0].device)
    for b in range(batch_size):
        l = length[b]
        x[b, :l] = seq[b][:l]
        x_mask[b, l:] = 1
    x_mask = x_mask > 0.5
    return x, x_mask


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.bn(self.linear(x))))


if __name__ == "__main__":
    from dataset import *

    dataset = ISLRDataSetV2()
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_func,
    )

    model = ISLRModel(embed_dim=256, n_head=4, ff_dim=256)

    for idx, batch in enumerate(train_loader):
        out = model(batch)
        print(out.shape)
        break
