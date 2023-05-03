import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from arcface import ArcMarginProduct

max_len = 80


# 2 encoder + more dropout layer + v2
class ISLRModelV6(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.1,
        max_len=384,
        n_labels=250,
        input_dim=912,
        n_layers=2,
        batch_first=True,
    ):
        super(ISLRModelV6, self).__init__()
        pos_encoding = positional_encoding(max_len, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))
        self.xyz_emb = nn.Sequential(nn.Linear(input_dim, embed_dim, bias=False))
        self.feature_extractor = _get_clones(
            TransformerBlockV2(
                embed_dim=embed_dim,
                n_head=n_head,
                ff_dim=ff_dim,
                dropout=dropout,
                batch_first=batch_first,
            ),
            N=n_layers,
        )
        self.logit = nn.Linear(embed_dim, n_labels)

        self.max_len = max_len
        self.input_dim = input_dim
        self.p = dropout

    def forward(self, batch):
        xyz = batch["xyz"]
        x, x_mask = pack_seqv2(xyz, self.max_len)
        B, L, _ = x.shape
        x = self.xyz_emb(x)
        x = F.dropout(
            x + self.pos_emb[:L].unsqueeze(0), p=self.p, training=self.training
        )
        x = torch.cat([self.cls_emb.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)

        for mod in self.feature_extractor:
            x = mod(x, x_mask)

        cls = x[:, 0]
        cls = F.dropout(cls, p=0.4, training=self.training)
        logit = self.logit(cls)

        return logit


# v6 + arcface
class ISLRModelArcFace(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.2,
        cls_dropout=0.2,
        max_len=64,
        n_labels=250,
        input_dim=1194,
        n_layers=5,
        s=32,
        m=0.1,
        k=3,
        batch_first=True,
    ):
        super(ISLRModelArcFace, self).__init__()
        pos_encoding = positional_encoding(max_len, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))
        self.xyz_emb = nn.Sequential(nn.Linear(input_dim, embed_dim, bias=False))
        self.feature_extractor = _get_clones(
            TransformerBlockV2(
                embed_dim=embed_dim,
                n_head=n_head,
                ff_dim=ff_dim,
                dropout=dropout,
                batch_first=batch_first,
            ),
            N=n_layers,
        )
        # self.logit = nn.Linear(embed_dim, n_labels)
        self.arc = ArcMarginProduct(embed_dim, n_labels, s=s, m=m, K=k)

        self.max_len = max_len
        self.input_dim = input_dim
        self.p = dropout
        self.cls_p = cls_dropout

    def forward(self, batch):
        xyz = batch["xyz"]
        if self.training:
            label = batch["label"]
        else:
            label = None

        x, x_mask = pack_seqv2(xyz, self.max_len)
        B, L, _ = x.shape
        x = self.xyz_emb(x)
        x = F.dropout(
            x + self.pos_emb[:L].unsqueeze(0), p=self.p, training=self.training
        )
        x = torch.cat([self.cls_emb.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)

        for mod in self.feature_extractor:
            x = mod(x, x_mask)

        cls = x[:, 0]
        cls = F.dropout(cls, p=self.cls_p, training=self.training)
        # logit = self.logit(cls)
        logit = self.arc(cls, label)

        return logit


# v6 + arcface + w/ simple c.e
class ISLRModelArcFaceCE(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.2,
        cls_dropout=0.2,
        max_len=64,
        n_labels=250,
        input_dim=1194,
        n_layers=5,
        s=32,
        m=0.1,
        k=3,
        batch_first=True,
    ):
        super(ISLRModelArcFaceCE, self).__init__()
        pos_encoding = positional_encoding(max_len, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))
        self.xyz_emb = nn.Sequential(nn.Linear(input_dim, embed_dim, bias=False))
        self.feature_extractor = _get_clones(
            TransformerBlockV2(
                embed_dim=embed_dim,
                n_head=n_head,
                ff_dim=ff_dim,
                dropout=dropout,
                batch_first=batch_first,
            ),
            N=n_layers,
        )
        self.logit = nn.Linear(embed_dim, n_labels)
        self.arc = ArcMarginProduct(embed_dim, n_labels, s=s, m=m, K=k)

        self.max_len = max_len
        self.input_dim = input_dim
        self.p = dropout
        self.cls_p = cls_dropout

    def forward(self, batch):
        xyz = batch["xyz"]
        if self.training:
            label = batch["label"]
        else:
            label = None

        x, x_mask = pack_seqv2(xyz, self.max_len)
        B, L, _ = x.shape
        x = self.xyz_emb(x)
        x = F.dropout(
            x + self.pos_emb[:L].unsqueeze(0), p=self.p, training=self.training
        )
        x = torch.cat([self.cls_emb.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)

        for mod in self.feature_extractor:
            x = mod(x, x_mask)

        cls = x[:, 0]
        cls = F.dropout(cls, p=self.cls_p, training=self.training)

        logit = self.logit(cls)
        arc_logit = self.arc(cls, label)

        return logit, arc_logit


# V6 + GCN embedding
from deprecated.gcn import GCNLayer
from deprecated.graph import Graph


class ISLRModelV8(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.1,
        max_len=384,
        n_labels=250,
        n_gcn_layers=2,
        gcn_c_in=2,
        gcn_c_out=4,
        n_tf_layers=5,
        graph="A",
        hop=1,
        batch_first=True,
    ):
        super(ISLRModelV8, self).__init__()
        pos_encoding = positional_encoding(max_len, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))

        xyz_emb = []
        c_in, c_out = gcn_c_in, gcn_c_out
        for i in range(n_gcn_layers - 1):
            xyz_emb += [
                GCNLayer(c_in=c_in, c_out=c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            c_in = c_out
        xyz_emb += [GCNLayer(c_in=c_in, c_out=c_out)]
        self.xyz_emb = ModuleList(xyz_emb)

        self.feature_extractor = _get_clones(
            TransformerBlockV2(
                embed_dim=embed_dim,
                n_head=n_head,
                ff_dim=ff_dim,
                dropout=dropout,
                batch_first=batch_first,
            ),
            N=n_tf_layers,
        )
        self.logit = nn.Linear(embed_dim, n_labels)

        self.max_len = max_len
        self.p = dropout
        if graph == "A":
            self.A = Graph(hop).A
        # elif graph == "B":
        #     self.A = GraphB().A

        self.num_point = self.A.shape[0]
        self.gcn_c_in = gcn_c_in

        assert embed_dim == gcn_c_out * self.num_point

    def forward(self, batch):
        xyz = batch["xyz"]
        x, x_mask = pack_seq(xyz, self.max_len, self.gcn_c_in)
        B, L, _, _ = x.shape
        A = self.A.repeat(B, L, 1, 1).cuda()

        for layer in self.xyz_emb:
            if isinstance(layer, GCNLayer):
                x = layer(x, A)
            else:
                x = layer(x)

        x = x.reshape(B, L, -1)
        x = F.dropout(
            x + self.pos_emb[:L].unsqueeze(0), p=self.p, training=self.training
        )
        x = torch.cat([self.cls_emb.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)

        for mod in self.feature_extractor:
            x = mod(x, x_mask)

        cls = x[:, 0]
        cls = F.dropout(cls, p=0.4, training=self.training)
        logit = self.logit(cls)

        return logit


class ISLRModelV8_1(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.1,
        window=64,
        n_labels=250,
        n_gcn_layers=2,
        gcn_c_in=58,
        gcn_c_out=256,
        n_tf_layers=5,
        graph="A",
        hop=2,
        batch_first=True,
    ):
        super(ISLRModelV8_1, self).__init__()
        pos_encoding = positional_encoding(window, embed_dim)
        self.pos_emb = nn.Parameter(pos_encoding)
        self.cls_emb = nn.Parameter(torch.zeros((1, embed_dim)))
        self.graph_norm = nn.LayerNorm(embed_dim)

        xyz_emb = []
        c_in, c_out = gcn_c_in, gcn_c_out
        for i in range(n_gcn_layers - 1):
            xyz_emb += [
                GCNLayer(c_in=c_in, c_out=c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            c_in = c_out
        xyz_emb += [GCNLayer(c_in=c_in, c_out=c_out)]
        self.xyz_emb = ModuleList(xyz_emb)

        self.feature_extractor = _get_clones(
            TransformerBlockV2(
                embed_dim=embed_dim,
                n_head=n_head,
                ff_dim=ff_dim,
                dropout=dropout,
                batch_first=batch_first,
            ),
            N=n_tf_layers,
        )
        self.logit = nn.Linear(embed_dim, n_labels)

        self.window = window
        self.p = dropout
        if graph == "A":
            self.A = Graph(hop).A
        # elif graph == "B":
        #     self.A = GraphB().A

        self.num_point = self.A.shape[0]
        self.gcn_c_in = gcn_c_in

    def forward(self, batch):
        xyz = batch["xyz"]
        x, x_mask = pack_seq(xyz, self.window, self.gcn_c_in)
        B, L, _, _ = x.shape
        A = self.A.repeat(B, L, 1, 1).cuda()

        for layer in self.xyz_emb:
            if isinstance(layer, GCNLayer):
                x = layer(x, A)
            else:
                x = layer(x)

        # average pooling
        x = x.mean(dim=2, keepdim=False)  # [B, L, 128]
        x = self.graph_norm(x)

        # x = x.reshape(B, L, -1)
        x = F.dropout(
            x + self.pos_emb[:L].unsqueeze(0), p=self.p, training=self.training
        )
        x = torch.cat([self.cls_emb.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)

        for mod in self.feature_extractor:
            x = mod(x, x_mask)

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

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x


class TransformerBlockV2(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        ff_dim,
        dropout=0.1,
        batch_first=True,
    ):
        super(TransformerBlockV2, self).__init__()
        self.attn = MultiHeadAttention(
            embed_dim, n_head, dropout=dropout, batch_first=batch_first
        )
        self.ffn = FeedForwardV2(embed_dim, ff_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(ff_dim)
        self.p = dropout

    def forward(self, x, x_mask=None):
        x = x + F.dropout(
            self.attn((self.norm1(x)), x_mask), p=self.p, training=self.training
        )
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


class FeedForwardV2(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.p = dropout

    def forward(self, x):
        return F.dropout(self.mlp(x), p=self.p, training=self.training)


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
    # x = x.reshape(batch_size, -1, K * point_dim)
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


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    from dataset import *
    from arcface import ArcMarginProduct

    dataset = ISLRDataSetV2(max_len=384, ver="xyzd_hdist_v2")
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_func,
    )

    model = ISLRModelV6(embed_dim=256, n_head=4, ff_dim=256)
    margin = ArcMarginProduct(250, 250)

    for idx, batch in enumerate(train_loader):
        y = batch["label"]
        print(y, y.shape)
        out = model(batch)
        print(out.shape)
        out = margin(out, y)
        print(out.shape)
        print(out)
        break
