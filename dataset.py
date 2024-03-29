import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from preproc import (
    preprocess,
    preproc_v0,
    preproc_v0_1,
    preproc_v0_2,
    preproc_v0_3,
    preproc_v0_4,
    preproc_v0_5,
    preproc_v0_6,
    preproc_v0_7,
    preproc_v0_8,
    preproc_v0_9,
    preproc_v0_91,
    preproc_v0_92,
    preproc_v0_93,
    preproc_v0_94,
    preproc_v1,
    preproc_v1_1,
    preproc_v1_2,
    preproc_v1_3,
)

from transforms import (
    random_noise,
    flip_x_hand,
    flip_x_lip,
    flip_x_spose,
    flip_x_eye,
    flip_x_nose,
    rotate,
    drop_landmark,
    interpolate,
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
        window=64,
        ver="base",
        indicies=None,
        flip_x=False,
        flip_x_v2=False,
        random_noise=False,
        rotate=False,
        drop_lm=False,
        interpolate=False,
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
        self.window = window
        self.ver = ver
        self.random_noise = random_noise
        self.flip_x = flip_x
        self.flip_x_v2 = flip_x_v2
        self.rotate = rotate
        self.drop_lm = drop_lm
        self.interpolate = interpolate

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
                if self.flip_x_v2:
                    xyz[:, :, :2] = xyz[:, :, :2] - 0.5

                lhand, rhand = xyz[:, LHAND], xyz[:, RHAND]
                lip = xyz[:, LIP]
                spose = xyz[:, SPOSE]
                leye = xyz[:, LEYE]
                reye = xyz[:, REYE]
                nose = xyz[:, NOSE]

                lhand, rhand = flip_x_hand(lhand, rhand)
                lip = flip_x_lip(lip)
                spose = flip_x_spose(spose)
                leye, reye = flip_x_eye(leye, reye)
                nose = flip_x_nose(nose)

                xyz[:, LHAND], xyz[:, RHAND] = lhand, rhand
                xyz[:, LIP] = lip
                xyz[:, SPOSE] = spose
                xyz[:, LEYE], xyz[:, REYE] = leye, reye
                xyz[:, NOSE] = nose

                if self.flip_x_v2:
                    xyz[:, :, :2] = xyz[:, :, :2] + 0.5

        if self.interpolate:
            if np.random.rand() < 0.5:
                ratio = (2 * np.random.random() - 1) * 0.25
                xyz = interpolate(xyz, ratio)

        if self.random_noise:
            if np.random.rand() < 0.5:
                xyz = random_noise(xyz)

        if self.drop_lm:
            if np.random.rand() < 0.5:
                lhand, rhand = xyz[:, LHAND], xyz[:, RHAND]
                lhand, rhand = drop_landmark(lhand, rhand, p=0.05)
                xyz[:, LHAND], xyz[:, RHAND] = lhand, rhand

        xyz = torch.from_numpy(xyz).float()
        if self.ver == "base":
            xyz = preprocess(xyz, self.max_len)
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
        elif self.ver == "v0_8":
            xyz = preproc_v0_8(xyz, self.max_len)
        elif self.ver == "v0_9":
            xyz = preproc_v0_9(xyz, self.max_len)
        elif self.ver == "v0_91":
            xyz = preproc_v0_91(xyz, self.max_len, self.window)
        elif self.ver == "v0_92":
            xyz = preproc_v0_92(xyz, self.max_len, self.window)
        elif self.ver == "v0_93":
            xyz = preproc_v0_93(xyz, self.max_len)
        elif self.ver == "v0_94":
            xyz = preproc_v0_94(xyz, self.max_len)
        elif self.ver == "v1":
            xyz = preproc_v1(xyz, self.max_len, self.window)
        elif self.ver == "v1_1":
            xyz = preproc_v1_1(xyz, self.max_len)
        elif self.ver == "v1_2":
            xyz = preproc_v1_2(xyz, self.max_len)
        elif self.ver == "v1_3":
            xyz = preproc_v1_3(xyz, self.max_len, window=self.window)
        else:
            raise NotImplementedError

        label = self.label_map[self.df.iloc[index].sign]

        d = {}
        d["xyz"] = xyz
        d["label"] = label

        return d


class ISLRDataSetV3(Dataset):
    def __init__(
        self,
        path="/sources/dataset",
        max_len=256,
        window=64,
        ver="v1",
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
        self.window = window
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
                lhand, rhand = xyz[:, LHAND], xyz[:, RHAND]
                lip = xyz[:, LIP]
                spose = xyz[:, SPOSE]
                leye = xyz[:, LEYE]
                reye = xyz[:, REYE]
                nose = xyz[:, NOSE]

                lhand, rhand = flip_x_hand(lhand, rhand)
                lip = flip_x_lip(lip)
                spose = flip_x_spose(spose)
                leye, reye = flip_x_eye(leye, reye)
                nose = flip_x_nose(nose)

                xyz[:, LHAND], xyz[:, RHAND] = lhand, rhand
                xyz[:, LIP] = lip
                xyz[:, SPOSE] = spose
                xyz[:, LEYE], xyz[:, REYE] = leye, reye
                xyz[:, NOSE] = nose

        if self.random_noise:
            if np.random.rand() < 0.5:
                xyz = random_noise(xyz)

        xyz = torch.from_numpy(xyz).float()
        if self.ver == "v1":
            xyz, mask = preproc_v1(xyz, self.max_len, self.window)
        else:
            raise NotImplementedError

        label = self.label_map[self.df.iloc[index].sign]

        return xyz, label, mask


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
POSE_OFFSET = 489
SPOSE = [POSE_OFFSET + i for i in [0, 1, 4, 11, 12, 13, 14, 15, 16, 23, 24]]
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
REYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
]
LEYE = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
NOSE = [1, 2, 98, 327]

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
