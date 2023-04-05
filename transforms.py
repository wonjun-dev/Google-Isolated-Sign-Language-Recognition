import numpy as np


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


def flip_x_spose(spose):
    spose[..., 0] *= -1
    spose = spose[:, [0, 2, 1, 4, 3, 6, 5, 8, 7]]
    return spose


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


def rotate(xyz, theta):
    radian = np.radians(theta)
    mat = np.array(
        [[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]]
    )
    xyz[:, :, :2] = xyz[:, :, :2] - 0.5
    xyz_reshape = xyz.reshape(-1, 2)
    xyz_rotate = np.dot(xyz_reshape, mat).reshape(xyz.shape)

    return xyz_rotate[:, :, :2] + 0.5


if __name__ == "__main__":
    import os
    from dataset import load_relevant_data_subset

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

    max_len = 384
    xyz = load_relevant_data_subset(
        "/sources/dataset/train_landmark_files/2044/635217.parquet"
    )
    print(xyz[0, 0])
    xyz = np.ones((1, 1, 3))
    print("---")
    print(
        rotate(
            xyz[
                :,
                :,
            ],
            90,
        )
    )
    # # print(xyz[0, :5])
    # xyz = random_noise(xyz)
    # # print(xyz[0, :5])

    # lhand, rhand = xyz[:, LHAND], xyz[:, RHAND]
    # slip = xyz[:, SLIP]
    # spose = xyz[:, POSE_SIM]
    # print(spose[0])

    # lhand, rhand = flip_x_hand(lhand, rhand)
    # slip = flip_x_slip(slip)
    # spose = flip_x_spose(spose)
    # print(spose[0])

    # print(xyz[:])
    # xyz[:, SLIP] = slip
    # xyz[:, LHAND] = lhand
    # xyz[:, POSE_SIM] = spose
    # xyz[:, RHAND] = rhand
    # print(xyz)
