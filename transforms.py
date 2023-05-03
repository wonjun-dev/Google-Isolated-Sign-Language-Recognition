import numpy as np
import torch.nn.functional as F
import torch


def random_noise(xyz, mean=0, std=1e-3):
    """
    Gaussian noise injection (mean: 0, std: 1e-3)
    Args:
        x: [None, # lm, 3 or 2]
    """
    size = xyz.shape
    gaussian = np.random.normal(mean, std, size)
    return xyz + gaussian


def flip_x_hand(lhand, rhand):
    rhand[..., 0] *= -1
    lhand[..., 0] *= -1
    rhand, lhand = lhand, rhand
    return lhand, rhand


def flip_x_slip(slip):
    slip[..., 0] *= -1
    slip = slip[
        :, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] + [19, 18, 17, 16, 15, 14, 13, 12, 11]
    ]
    return slip


def flip_x_lip(lip):
    lip[..., 0] *= -1
    lip = lip[
        :,
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        + [19, 18, 17, 16, 15, 14, 13, 12, 11]
        + [29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
        + [39, 38, 37, 36, 35, 34, 33, 32, 31, 30],
    ]
    return lip


def flip_x_spose(spose):
    spose[..., 0] *= -1
    spose = spose[:, [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9]]
    return spose


def flip_x_eye(leye, reye):
    reye[..., 0] *= -1
    leye[..., 0] *= -1
    reye, leye = leye, reye
    return leye, reye


def flip_x_nose(nose):
    nose[..., 0] *= -1
    nose = nose[:, [0, 1, 3, 2]]
    return nose


def rotate(xyz, theta):
    radian = np.radians(theta)
    mat = np.array(
        [[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]]
    )
    xyz[:, :, :2] = xyz[:, :, :2] - 0.5
    xyz_reshape = xyz.reshape(-1, 2)
    xyz_rotate = np.dot(xyz_reshape, mat).reshape(xyz.shape)

    return xyz_rotate[:, :, :2] + 0.5


def drop_landmark(lhand, rhand, p=0.05):
    L = lhand.shape[0]
    mask = np.random.choice([0, 1], size=(L, 21, 1), p=[p, 1 - p])
    lhand *= mask
    rhand *= mask
    return lhand, rhand


def interpolate(xyz, ratio):
    L, V, C = xyz.shape
    xyz = torch.from_numpy(xyz)
    xyz = xyz.permute(1, 2, 0).contiguous().view(V * C, L)
    xyz = xyz[None, None, :, :]

    resize = int(L * (1 + ratio))

    xyz = F.interpolate(
        xyz, size=(V * C, resize), mode="bilinear", align_corners=False
    ).squeeze()
    xyz = xyz.view(V, C, -1).permute(2, 0, 1).contiguous()  # [L, V, C]

    return xyz.numpy()


if __name__ == "__main__":
    import os
    from dataset import load_relevant_data_subset

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
    LH_OFFSET = 468
    LHAND = [LH_OFFSET + i for i in range(21)]
    RH_OFFSET = 522
    RHAND = [RH_OFFSET + i for i in range(21)]

    POSE_OFFSET = 489
    SPOSE = [POSE_OFFSET + i for i in [0, 1, 4, 11, 12, 13, 14, 15, 16, 23, 24]]
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

    max_len = 384
    xyz = load_relevant_data_subset(
        "/sources/dataset/train_landmark_files/2044/635217.parquet"
    )
    # lhand, rhand = xyz[:, LHAND], xyz[:, RHAND]
    # print(lhand, rhand)
    # lhand, rhand = drop_landmark(lhand, rhand)
    # print("---")
    # print(lhand, rhand)

    print(xyz.shape)
    print(xyz[0])
    xyz = interpolate(xyz, -0.20)
    print(xyz.shape)
    print(xyz[0])
