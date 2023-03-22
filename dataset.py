import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from preproc import (
    preprocess,
    preprocess_centercrop,
    preprocess_body,
    preprocess_bodyarm,
    preprocess_bodyarm2,
    preprocess_wonorm,
    preprocess_xyzd,
    preprocess_xyzd_hdist,
    preprocess_xyzd_hdist_interpolate,
    preprocess_xyzd_hdist_hdistd,
    preprocess_xyzd_hdist_nl,
    preprocess_xyzd_hdist_pnv,
    preprocess_xyzd_hdist_pnv_nl,
    preprocess_bm0,
    preprocess_bm0x5,
    preprocess_smooth,
)

import time


class ISLRDataSet(Dataset):
    def __init__(self, path="/sources/dataset", ver=0, indicies=None):
        data_path = os.path.join(path, "features", f"ver{ver}")

        features = np.load(os.path.join(data_path, "feature_data.npy"))
        labels = np.load(os.path.join(data_path, "feature_labels.npy"))

        if indicies is not None:
            self.features = features[indicies]
            self.labels = labels[indicies]
        else:
            self.features = features
            self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.features[index, :], dtype=torch.float32), self.labels[
            index
        ].astype(int)


class ISLRDataSetV2(Dataset):
    def __init__(self, path="/sources/dataset", max_len=80, ver="base", indicies=None):
        self.path = path
        df = pd.read_csv(os.path.join(path, "train.csv"))
        self.label_map = json.load(
            open(os.path.join(path, "sign_to_prediction_index_map.json"))
        )

        if indicies is not None:
            self.df = df.iloc[indicies]
            self.df = self.df.reset_index(drop=True)
        else:
            self.df = df

        self.max_len = max_len
        self.ver = ver

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xyz = load_relevant_data_subset(
            os.path.join(self.path, self.df.iloc[index].path)
        )
        xyz = torch.from_numpy(xyz).float()
        if self.ver == "base":
            xyz = preprocess(xyz, self.max_len)
        elif self.ver == "base_centercrop":
            xyz = preprocess_centercrop(xyz, self.max_len)
        elif self.ver == "body":
            xyz = preprocess_body(xyz, self.max_len)
        elif self.ver == "bodyarm":
            xyz = preprocess_bodyarm(xyz, self.max_len)
        elif self.ver == "bodyarm2":
            xyz = preprocess_bodyarm2(xyz, self.max_len)
        elif self.ver == "wonrom_base":
            xyz = preprocess_wonorm(xyz, self.max_len)
        elif self.ver == "xyzd":
            xyz = preprocess_xyzd(xyz, self.max_len)
        elif self.ver == "xyzd_hdist":
            xyz = preprocess_xyzd_hdist(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_interp":
            xyz = preprocess_xyzd_hdist_interpolate(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_hdistd":
            xyz = preprocess_xyzd_hdist_hdistd(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_nl":
            xyz = preprocess_xyzd_hdist_nl(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_pnv":
            xyz = preprocess_xyzd_hdist_pnv(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_pnv_nl":
            xyz = preprocess_xyzd_hdist_pnv_nl(xyz, self.max_len)
        elif self.ver == "bm0":
            xyz = preprocess_bm0(xyz, self.max_len)
        elif self.ver == "bm0x5":
            xyz = preprocess_bm0x5(xyz, self.max_len)
        else:
            raise NotImplementedError

        label = self.label_map[self.df.iloc[index].sign]

        d = {}
        d["xyz"] = xyz
        d["label"] = label

        return d


def collate_func(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    d["label"] = torch.LongTensor(d["label"])
    return d


ROWS_PER_FRAME = 543


def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


if __name__ == "__main__":
    dataset = ISLRDataSetV2()
    print(dataset.__len__())
    s = time.time()
    item = dataset.__getitem__(0)
    print(time.time() - s)
    # print(item)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_func,
    )

    s = time.time()
    for idx, item in enumerate(train_loader):
        print(time.time() - s)
        print(len(item["xyz"]))
        break
