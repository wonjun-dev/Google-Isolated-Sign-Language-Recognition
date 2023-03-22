import torch
import numpy as np
from scipy.interpolate import interp1d

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

POSE_OFFSET = 489
BODY = [500, 501, 512, 513]  #  [11, 12, 23, 24]
ARM = [
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    509,
    510,
    511,
]  # [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

LH_OFFSET = 468
LHAND = [LH_OFFSET + i for i in range(21)]
RH_OFFSET = 522
RHAND = [RH_OFFSET + i for i in range(21)]


def preprocess_wonorm(xyz, max_len):
    # xyz = xyz - xyz[~torch.isnan(xyz)].mean(
    #     0, keepdims=True
    # )  # noramlisation to common maen
    # xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


def preprocess(xyz, max_len):
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


def preprocess_centercrop(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]

    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    return xyz


def preprocess_body(xyz, max_len):
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    body = xyz[:, BODY]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            body,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


def preprocess_bodyarm(xyz, max_len):
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    body = xyz[:, BODY]
    arm = xyz[:, ARM]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            body,
            arm,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


def preprocess_bodyarm2(xyz, max_len):
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    body = xyz[:, BODY]
    arm = xyz[:, ARM[:4]]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            body,
            arm,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


def preprocess_xyzd(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]

    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)

    xyz = xyz[:, LIP + LHAND + RHAND]

    # (dim2_left, dim2_right, dim1_top, dim1_bottom, dim0_left, dim0_right)
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    x = torch.cat(
        [  # (none, 82, 6)
            xyz,
            torch.from_numpy(dxyz),
        ],
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_xyzd_hdist(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    # hand joint-wise distance
    mask = torch.tril(torch.ones(L, 21, 21, dtype=torch.bool), diagonal=-1)
    lhand = xyz[:, :21, :2]
    ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rhand = xyz[:, 21:42, :2]
    rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            torch.from_numpy(dxyz).reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 912)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_xyzd_hdist_interpolate(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # interpolate
    f = interp1d(np.arange(0, L), xyz, axis=0)
    xyz = torch.from_numpy(f(np.arange(0, L - 1, 0.5)))
    L = len(xyz)

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    # hand joint-wise distance
    mask = torch.tril(torch.ones(L, 21, 21, dtype=torch.bool), diagonal=-1)
    lhand = xyz[:, :21, :2]
    ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rhand = xyz[:, 21:42, :2]
    rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            torch.from_numpy(dxyz).reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 912)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_xyzd_hdist_hdistd(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    # hand joint-wise distance
    mask = torch.tril(torch.ones(L, 21, 21, dtype=torch.bool), diagonal=-1)
    lhand = xyz[:, :21, :2]
    ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)
    ld = ld.reshape(L, -1)

    rhand = xyz[:, 21:42, :2]
    rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)
    rd = rd.reshape(L, -1)

    # hdist motion
    dld = ld[:-1] - ld[1:]
    dld = np.pad(dld, [[0, 1], [0, 0]])

    drd = rd[:-1] - rd[1:]
    drd = np.pad(drd, [[0, 1], [0, 0]])

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            torch.from_numpy(dxyz).reshape(L, -1),
            ld,
            rd,
            torch.from_numpy(dld),
            torch.from_numpy(drd),
        ],  # (none, 1332)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_xyzd_hdist_nl(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    # hand joint-wise distance
    mask = torch.tril(torch.ones(L, 21, 21, dtype=torch.bool), diagonal=-1)
    lhand = xyz[:, :21, :2]
    ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rhand = xyz[:, 21:42, :2]
    rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            torch.from_numpy(dxyz).reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 912)
        -1,
    )
    # noramlization
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)

    x[torch.isnan(x)] = 0
    return x


def preprocess_xyzd_hdist_pnv(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    # hand joint-wise distance
    mask = torch.tril(torch.ones(L, 21, 21, dtype=torch.bool), diagonal=-1)
    lhand = xyz[:, :21, :2]
    ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rhand = xyz[:, 21:42, :2]
    rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    # palm unit normal vector
    lpalm = xyz[:, [0, 5, 17]]
    lpnv = np.cross(lpalm[:, 2] - lpalm[:, 0], lpalm[:, 1] - lpalm[:, 0])
    lnorm = np.linalg.norm(lpnv, axis=-1, keepdims=True)
    lpnv /= lnorm

    rpalm = xyz[:, [21, 26, 38]]
    rpnv = np.cross(rpalm[:, 2] - rpalm[:, 0], rpalm[:, 1] - rpalm[:, 0])
    rnorm = np.linalg.norm(rpnv, axis=-1, keepdims=True)
    rpnv /= rnorm

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            torch.from_numpy(dxyz).reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            torch.from_numpy(lpnv),
            torch.from_numpy(rpnv),
        ],  # (none, 918)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_xyzd_hdist_pnv_nl(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    # hand joint-wise distance
    mask = torch.tril(torch.ones(L, 21, 21, dtype=torch.bool), diagonal=-1)
    lhand = xyz[:, :21, :2]
    ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rhand = xyz[:, 21:42, :2]
    rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    # palm unit normal vector
    lpalm = xyz[:, [0, 5, 17]]
    lpnv = np.cross(lpalm[:, 2] - lpalm[:, 0], lpalm[:, 1] - lpalm[:, 0])
    lnorm = np.linalg.norm(lpnv, axis=-1, keepdims=True)
    lpnv /= lnorm

    rpalm = xyz[:, [21, 26, 38]]
    rpnv = np.cross(rpalm[:, 2] - rpalm[:, 0], rpalm[:, 1] - rpalm[:, 0])
    rnorm = np.linalg.norm(rpnv, axis=-1, keepdims=True)
    rpnv /= rnorm

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            torch.from_numpy(dxyz).reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            torch.from_numpy(lpnv),
            torch.from_numpy(rpnv),
        ],  # (none, 918)
        -1,
    )

    # noramlization
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)

    x[torch.isnan(x)] = 0
    return x


def preprocess_bm0(xyz, max_len):
    bm = np.expand_dims(xyz[:, POSE_OFFSET], axis=1)
    xyz = xyz - bm

    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


def preprocess_bm0x5(xyz, max_len):
    bm = np.expand_dims(xyz[:, POSE_OFFSET], axis=1)
    xyz = (xyz - bm) * 5

    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


def preprocess_smooth(xyz, max_len):
    xyz = xyz - xyz[~torch.isnan(xyz)].mean(
        0, keepdims=True
    )  # noramlisation to common maen
    xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    lip = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat(
        [  # (none, 82, 3)
            lip,
            lhand,
            rhand,
        ],
        1,
    )
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_len]
    return xyz


if __name__ == "__main__":
    import os
    from dataset import load_relevant_data_subset

    max_len = 80
    xyz = load_relevant_data_subset(
        "/sources/dataset/train_landmark_files/2044/635217.parquet"
    )
    xyz = torch.from_numpy(xyz).float()
    xyz = preprocess_xyzd_hdist_interpolate(xyz, max_len)
    print(min(xyz[5]))
    print(max(xyz[5]))
    print(xyz.shape)
