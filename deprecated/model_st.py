import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList
import copy


class MultiHeadSelfAttn(nn.Module):
    def __init__(self, in_dim, out_dim, num_head=2, dropout=0.1, groups=1) -> None:
        super(MultiHeadSelfAttn, self).__init__()
        self.out_dim = out_dim
        self.num_head = num_head
        self.head_dim = int(out_dim // num_head)
        self.scale = self.head_dim**-0.5
        assert self.out_dim == self.head_dim * num_head

        self.wk = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.wq = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.wv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(
            out_dim, out_dim, kernel_size=1, groups=groups
        )  # TODO  groups
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # input: [N, C, T, V]
        k, q, v = self.wk(x), self.wq(x), self.wv(x)
        k, q, v = self.split_heads(k), self.split_heads(q), self.split_heads(v)
        x = self.attention(q, k, v, mask)
        return x

    def split_heads(self, x):
        N, C, T, V = x.shape
        x = x.view(N, self.num_head, self.head_dim, T, V).permute(0, 3, 1, 4, 2)
        return x

    def attention(self, q, k, v, mask=None):
        # input: [N, T, H, V, C]
        # output: [N, C, T, V]
        N, T, H, V, C = q.shape
        score = torch.matmul(q, k.transpose(-2, -1))
        score = score / self.scale

        if mask is not None:
            # N T V
            mask = mask.unsqueeze(2).unsqueeze(3)  # N T 1 1 V
            score = score.masked_fill(mask == 1, -1e10)

        score = F.softmax(score, dim=-1)
        attn_out = torch.matmul(self.attn_dropout(score), v)
        attn_out = attn_out.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        return self.proj_dropout(self.proj(attn_out))


class TemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SpatialAttn(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, dropout, groups) -> None:
        super(SpatialAttn, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.attn = MultiHeadSelfAttn(in_dim, out_dim, num_head, dropout, groups)

    def forward(self, x, mask=None):
        x = self.attn(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), mask)
        return x


class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        dilations=[1, 2, 3, 4],
    ) -> None:
        super(MultiScaleTemporalConv, self).__init__()
        assert out_ch % (len(dilations) + 2) == 0

        self.num_branches = len(dilations) + 2
        branch_ch = out_ch // self.num_branches
        kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, branch_ch, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_ch),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_ch,
                        branch_ch,
                        kernel_size=ks,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
                for ks, dilation in zip(kernel_size, dilations)
            ]
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_ch),
            )
        )
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_ch, branch_ch, kernel_size=1, padding=0, stride=(stride, 1)
                ),
                nn.BatchNorm2d(branch_ch),
            )
        )

        self.residual = lambda x: 0
        self.apply(weights_init)

    def forward(self, x):
        # input : [N, C, T, V]
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class STBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        dropout=0.1,
        groups=1,
        kernel_size=3,
        dilations=[1, 2],
        num_head=4,
        residual=True,
    ) -> None:
        super(STBlock, self).__init__()
        self.sn = SpatialAttn(in_ch, out_ch, num_head, dropout, groups)
        self.tn = MultiScaleTemporalConv(
            out_ch, out_ch, kernel_size=kernel_size, stride=stride, dilations=dilations
        )

        self.activation = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(1, 1),
                    padding=(0, 0),
                    stride=1,
                    bias=False,
                ),
            )

    def forward(self, x, mask=None):
        out = self.activation(self.tn(self.sn(x, mask))) + self.residual(x)
        return out


class JointEmbedding(nn.Module):
    def __init__(self, joint_dim, emb_dim):
        super(JointEmbedding, self).__init__()
        self.embedding = nn.Conv2d(
            joint_dim, emb_dim, kernel_size=(1, 1), padding=(0, 0), stride=1, bias=False
        )

    def forward(self, x):
        # input [N, 2, T, V]
        return self.embedding(x)


class STModel(nn.Module):
    def __init__(
        self,
        n_layers=6,
        in_ch=2,
        dropout=0.1,
        num_head=4,
        kernel_size=3,
        dilations=[1, 2],
        num_point=53,
        num_class=250,
    ):
        super(STModel, self).__init__()
        self.num_head = num_head
        self.num_point = num_point
        self.num_class = num_class
        # self.data_bn = nn.BatchNorm1d(in_ch * num_point)
        self.emb_layer = JointEmbedding(joint_dim=in_ch, emb_dim=24 * num_head)
        self.pos_encoding = positional_encoding(num_point, 24 * num_head)
        self.pos_emb = nn.Parameter(self.pos_encoding)
        self.feature_extractor = _get_clones(
            STBlock(
                24 * num_head,
                24 * num_head,
                num_head=num_head,
                kernel_size=kernel_size,
                dilations=dilations,
            ),
            N=n_layers,
        )

        self.fc = nn.Linear(24 * num_head, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [N, C, T, V], mask: [N, T, V]
        N, C, T, V = x.shape
        # for data bn
        # x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, V, C, T).contiguous().permute(0, 2, 3, 1)

        x = self.emb_layer(x)
        x = F.dropout(
            x + self.pos_emb.unsqueeze(0).unsqueeze(2), p=0.1, training=self.training
        )
        for mod in self.feature_extractor:
            x = mod(x, mask)

        N, C, T, V = x.shape
        x = x * (1 - mask.unsqueeze(1))
        x = x.mean(dim=[2, 3])
        # x = x.view(N, C, -1)
        # x = x.mean(-1)
        x = self.fc(self.dropout(x))
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if (
            hasattr(m, "bias")
            and m.bias is not None
            and isinstance(m.bias, torch.Tensor)
        ):
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


def positional_encoding(length, embed_dim):
    dim = embed_dim // 2
    position = np.arange(length)[:, np.newaxis]  # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :] / dim  # (1, dim)
    angle = 1 / (10000**dim)  # (1, dim)
    angle = position * angle  # (pos, dim)
    pos_embed = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed.transpose(1, 0)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    from dataset import load_relevant_data_subset
    import torch
    from preproc import preproc_v1

    xyz = load_relevant_data_subset(
        "/sources/dataset/train_landmark_files/2044/635217.parquet"
    )

    xyz = torch.from_numpy(xyz).float()
    xyz = preproc_v1(xyz)
    xyz = xyz.unsqueeze(0)
    print(xyz.shape)
    # attn = STBlock(2, 20)
    attn = STModel(2)
    print(attn(xyz).shape)
