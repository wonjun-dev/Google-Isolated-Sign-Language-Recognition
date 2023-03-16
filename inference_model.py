import torch
import torch.nn as nn
import torch.nn.functional as F

from model import FeedForward, positional_encoding

max_length = 80
embed_dim = 256
n_head = 4
ff_dim = 256


class InferenceMultiHeadAttention(nn.Module):
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

    def forward(self, x):
        q = F.linear(
            x[:1],
            self.mha.in_proj_weight[:embed_dim],
            self.mha.in_proj_bias[:embed_dim],
        )  # since we need only cls
        k = F.linear(
            x,
            self.mha.in_proj_weight[embed_dim : embed_dim * 2],
            self.mha.in_proj_bias[embed_dim : embed_dim * 2],
        )
        v = F.linear(
            x,
            self.mha.in_proj_weight[embed_dim * 2 :],
            self.mha.in_proj_bias[embed_dim * 2 :],
        )
        q = q.reshape(-1, n_head, embed_dim // n_head).permute(1, 0, 2)
        k = k.reshape(-1, n_head, embed_dim // n_head).permute(1, 2, 0)
        v = v.reshape(-1, n_head, embed_dim // n_head).permute(1, 0, 2)
        dot = torch.matmul(q, k) * (1 / (embed_dim // n_head) ** 0.5)  # H L L
        attn = F.softmax(dot, -1)  #   L L
        out = torch.matmul(attn, v)  #   L H dim
        out = out.permute(1, 0, 2).reshape(-1, embed_dim)
        out = F.linear(out, self.mha.out_proj.weight, self.mha.out_proj.bias)
        return out


class InferenceTransformerBlock(nn.Module):
    def __init__(
        self, embed_dim, n_head, ff_dim, dropout=0.1, n_layer=1, batch_first=True
    ):
        super().__init__()
        self.attn = InferenceMultiHeadAttention(
            embed_dim, n_head, dropout=dropout, batch_first=batch_first
        )
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(ff_dim)

    def forward(self, x):
        x = x + self.attn((self.norm1(x)))
        x = x + self.ffn((self.norm2(x)))
        return x


class ISLRInferenceModel(nn.Module):
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
        batch_first=True,
    ):
        super().__init__()
        pos_encoding = positional_encoding(max_len, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))
        self.xyz_emb = nn.Sequential(nn.Linear(num_points * 3, embed_dim, bias=False))
        self.feature_extractor = InferenceTransformerBlock(
            embed_dim=embed_dim, n_head=n_head, ff_dim=ff_dim
        )
        self.logit = nn.Linear(embed_dim, n_labels)

    def forward(self, xyz):
        L = xyz.shape[0]
        x_embed = self.xyz_emb(xyz.flatten(1))
        x = x_embed[:L] + self.pos_emb[:L]
        x = torch.cat([self.cls_emb, x], 0)
        x = self.feature_extractor(x)
        cls = x[[0]]
        logit = self.logit(cls)
        return logit
