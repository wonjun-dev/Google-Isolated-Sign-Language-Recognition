import copy

import torch
import torch.nn as nn
from torch.nn import ModuleList

from feat_gen import FeatureGen, load_relevant_data_subset


class TFLiteModel(nn.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.feat_gen = FeatureGen()
        self.model = model

    def forward(self, x):
        x = self.feat_gen(x)
        x = self.model(x)
        return x


class ISLRModel(nn.Module):
    def __init__(self, encoder_layer, n_labels=250):
        super(ISLRModel, self).__init__()
        self.encoder1 = encoder_layer(3150, 3150)
        self.encoder2 = encoder_layer(3150, 1000)
        self.classifer = nn.Linear(1000, n_labels)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        return self.classifer(x)


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.bn(self.linear(x))))


if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd

    ROOT_PATH = "/sources/dataset"

    model = TFLiteModel(MLPLayer)

    # Inference mode
    model.eval()
    df = pd.read_csv(os.path.join(ROOT_PATH, "train.csv"))
    x = load_relevant_data_subset(os.path.join(ROOT_PATH, df.iloc[0].path))
    out = model(x)
    print(out)

    # Training mode
    from dataset import ISLRDataSet

    model.train()
    dataset = ISLRDataSet()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
    for x, y in dataloader:
        out = model(x)
        print(out)
        break
