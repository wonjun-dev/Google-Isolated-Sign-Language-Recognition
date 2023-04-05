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
    preprocess_xyzd_hdist_v2,
    preprocess_xyzd_hdist_v3,
    preprocess_xyzd_hdist_v4,
    preprocess_xyzd_hdist_v5,
    preprocess_xyzd_hdist_v6,
    preprocess_xyzd_hdist_interpolate,
    preprocess_xyzd_hdist_hdistd,
    preprocess_xyzd_hdist_nl,
    preprocess_xyzd_hdist_pnv,
    preprocess_xyzd_hdist_pnv_nl,
    preprocess_bm0,
    preprocess_bm0x5,
    preprocess_smooth,
    preprocess_v1,
    preprocess_v1_1,
    preprocess_v2,
    preprocess_v3,
    preprocess_v4,
    preprocess_v5,
    preprocess_v6,
    preprocess_v7,
    preprocess_v8,
    preprocess_v8_1,
    preprocess_v9,
    preprocess_v9_1,
    preproc_v0,
    preproc_v0_1,
    preproc_v0_2,
    preproc_v0_3,
    preproc_v0_4,
    preproc_v0_5,
    preproc_v0_6,
    preproc_v0_7,
    preproc_v1,
)

from transforms import (
    random_noise,
    flip_x_hand,
    flip_x_slip,
    flip_x_lip,
    flip_x_spose,
    rotate,
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
    def __init__(
        self,
        path="/sources/dataset",
        max_len=80,
        ver="base",
        indicies=None,
        flip_x=False,
        random_noise=False,
        rotate=False,
    ):
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
        self.random_noise = random_noise
        self.flip_x = flip_x
        self.rotate = rotate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xyz = load_relevant_data_subset(
            os.path.join(self.path, self.df.iloc[index].path)
        )
        if self.rotate:
            if np.random.rand() < 0.5:
                angle = (2 * np.random.random() - 1) * 13
                xyz[:, :, :2] = rotate(xyz[:, :, :2], angle)

        if self.flip_x:
            if np.random.rand() < 0.5:
                if self.ver == "v0":
                    lhand, rhand = xyz[:, LHAND], xyz[:, RHAND]
                    lip = xyz[:, LIP]

                    lhand, rhand = flip_x_hand(lhand, rhand)
                    lip = flip_x_lip(lip)

                    xyz[:, LHAND], xyz[:, RHAND] = lhand, rhand
                    xyz[:, LIP] = lip

                else:
                    lhand, rhand = xyz[:, LHAND], xyz[:, RHAND]
                    slip = xyz[:, SLIP]
                    spose = xyz[:, POSE_SIM]

                    lhand, rhand = flip_x_hand(lhand, rhand)
                    slip = flip_x_slip(slip)
                    spose = flip_x_spose(spose)

                    xyz[:, LHAND], xyz[:, RHAND] = lhand, rhand
                    xyz[:, SLIP] = slip
                    xyz[:, POSE_SIM] = spose

        if self.random_noise:
            if np.random.rand() < 0.5:
                xyz = random_noise(xyz)

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
        elif self.ver == "xyzd_hdist_v2":
            xyz = preprocess_xyzd_hdist_v2(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_v3":
            xyz = preprocess_xyzd_hdist_v3(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_v4":
            xyz = preprocess_xyzd_hdist_v4(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_v5":
            xyz = preprocess_xyzd_hdist_v5(xyz, self.max_len)
        elif self.ver == "xyzd_hdist_v6":
            xyz = preprocess_xyzd_hdist_v6(xyz, self.max_len)
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
        elif self.ver == "preproc_v1":
            xyz = preprocess_v1(xyz, self.max_len)
        elif self.ver == "preproc_v1_1":
            xyz = preprocess_v1_1(xyz, self.max_len)
        elif self.ver == "preproc_v2":
            xyz = preprocess_v2(xyz, self.max_len)
        elif self.ver == "preproc_v3":
            xyz = preprocess_v3(xyz, self.max_len)
        elif self.ver == "preproc_v4":
            xyz = preprocess_v4(xyz, self.max_len)
        elif self.ver == "preproc_v5":
            xyz = preprocess_v5(xyz, self.max_len)
        elif self.ver == "preproc_v6":
            xyz = preprocess_v6(xyz, self.max_len)
        elif self.ver == "preproc_v7":
            xyz = preprocess_v7(xyz, self.max_len)
        elif self.ver == "preproc_v8":
            xyz = preprocess_v8(xyz, self.max_len)
        elif self.ver == "preproc_v8_1":
            xyz = preprocess_v8_1(xyz, self.max_len)
        elif self.ver == "preproc_v9":
            xyz = preprocess_v9(xyz, self.max_len)
        elif self.ver == "preproc_v9_1":
            xyz = preprocess_v9_1(xyz, self.max_len)
        elif self.ver == "v0":
            xyz = preproc_v0(xyz, self.max_len)
        elif self.ver == "v0_1":
            xyz = preproc_v0_1(xyz, self.max_len)
        elif self.ver == "v0_2":
            xyz = preproc_v0_2(xyz, self.max_len)
        elif self.ver == "v0_3":
            xyz = preproc_v0_3(xyz, self.max_len)
        elif self.ver == "v0_4":
            xyz = preproc_v0_4(xyz, self.max_len)
        elif self.ver == "v0_5":
            xyz = preproc_v0_5(xyz, self.max_len)
        elif self.ver == "v0_6":
            xyz = preproc_v0_6(xyz, self.max_len)
        elif self.ver == "v0_7":
            xyz = preproc_v0_7(xyz, self.max_len)
        elif self.ver == "v1":
            xyz = preproc_v1(xyz, self.max_len)
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


LH_OFFSET = 468
LHAND = [LH_OFFSET + i for i in range(21)]
RH_OFFSET = 522
RHAND = [RH_OFFSET + i for i in range(21)]
simple_pose = [0, 1, 4, 11, 12, 13, 14, 15, 16]
POSE_SIM = [489 + i for i in simple_pose]
SLIP = [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
]
LIP = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]

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
