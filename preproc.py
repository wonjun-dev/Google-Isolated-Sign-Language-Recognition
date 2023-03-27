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
BODY = [
    489,
    490,
    491,
    492,
    493,
    494,
    495,
    496,
    497,
    498,
    499,
    500,
    501,
    512,
    513,
]  #  [0~10, 11, 12, 23, 24]
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
POSE = BODY + ARM[:4]

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


def preprocess_xyzd_hdist_v2(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    valid_mask = ~torch.isnan(xyz)
    sum_xyz = torch.nansum(xyz, dim=(0, 1))
    count_xyz = torch.sum(valid_mask, dim=(0, 1)).to(torch.float32)
    mean_xyz = sum_xyz / count_xyz

    gap = torch.where(valid_mask, xyz - mean_xyz, torch.tensor(float("nan")))
    sum_sq_diff = torch.nansum(gap**2, dim=(0, 1))
    std_xyz = torch.sqrt(sum_sq_diff / count_xyz)

    xyz = torch.where(valid_mask, gap / std_xyz, torch.tensor(float("nan")))

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


def preprocess_xyzd_hdist_v3(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    valid_mask = ~torch.isnan(xyz)
    sum_xyz = torch.nansum(xyz, dim=(0))
    count_xyz = torch.sum(valid_mask, dim=(0)).to(torch.float32)
    mean_xyz = sum_xyz / count_xyz

    gap = torch.where(valid_mask, xyz - mean_xyz, torch.tensor(float("nan")))
    sum_sq_diff = torch.nansum(gap**2, dim=(0))
    std_xyz = torch.sqrt(sum_sq_diff / count_xyz)

    xyz = torch.where(valid_mask, gap / std_xyz, torch.tensor(float("nan")))

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


def preprocess_xyzd_hdist_v4(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    valid_mask = ~torch.isnan(xyz)
    mean_xyz = torch.tensor([0.4706, 0.4606, -0.0453])
    std_xyz = torch.tensor([0.1036, 0.2405, 0.3024])

    gap = torch.where(valid_mask, xyz - mean_xyz, torch.tensor(float("nan")))

    xyz = torch.where(valid_mask, gap / std_xyz, torch.tensor(float("nan")))

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


def preprocess_xyzd_hdist_v5(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    valid_mask = ~torch.isnan(xyz)

    sum_xy = torch.nansum(xyz[:, :, :2], dim=(0, 1))
    sum_lip_z = torch.nansum(xyz[:, LIP, -1], dim=(0, 1))
    sum_lhand_z = torch.nansum(xyz[:, LHAND, -1], dim=(0, 1))
    sum_rhand_z = torch.nansum(xyz[:, RHAND, -1], dim=(0, 1))

    count_xy = torch.sum(valid_mask[:, :, :2], dim=(0, 1)).to(torch.float32)
    count_lip_z = torch.sum(valid_mask[:, LIP, -1], dim=(0, 1)).to(torch.float32)
    count_lhand_z = torch.sum(valid_mask[:, LHAND, -1], dim=(0, 1)).to(torch.float32)
    count_rhand_z = torch.sum(valid_mask[:, RHAND, -1], dim=(0, 1)).to(torch.float32)

    mean_xy = sum_xy / count_xy
    mean_lip_z = sum_lip_z / count_lip_z
    mean_lhand_z = sum_lhand_z / count_lhand_z
    mean_rhand_z = sum_rhand_z / count_rhand_z

    gap_xy = torch.where(
        valid_mask[:, :, :2], xyz[:, :, :2] - mean_xy, torch.tensor(float("nan"))
    )
    sum_sq_diff_xy = torch.nansum(gap_xy**2, dim=(0, 1))
    std_xy = torch.sqrt(sum_sq_diff_xy / count_xy)

    gap_lip_z = torch.where(
        valid_mask[:, LIP, -1], xyz[:, LIP, -1] - mean_lip_z, torch.tensor(float("nan"))
    )
    gap_lhand_z = torch.where(
        valid_mask[:, LHAND, -1],
        xyz[:, LHAND, -1] - mean_lhand_z,
        torch.tensor(float("nan")),
    )
    gap_rhand_z = torch.where(
        valid_mask[:, RHAND, -1],
        xyz[:, RHAND, -1] - mean_rhand_z,
        torch.tensor(float("nan")),
    )
    sum_sq_diff_lip_z = torch.nansum(gap_lip_z**2, dim=(0, 1))
    sum_sq_diff_lhand_z = torch.nansum(gap_lhand_z**2, dim=(0, 1))
    sum_sq_diff_rhand_z = torch.nansum(gap_rhand_z**2, dim=(0, 1))
    std_lip_z = torch.sqrt(sum_sq_diff_lip_z / count_lip_z)
    std_lhand_z = torch.sqrt(sum_sq_diff_lhand_z / count_lhand_z)
    std_rhand_z = torch.sqrt(sum_sq_diff_rhand_z / count_rhand_z)

    norm_xy = torch.where(
        valid_mask[:, :, :2], gap_xy / std_xy, torch.tensor(float("nan"))
    )
    norm_lip_z = torch.where(
        valid_mask[:, LIP, -1], gap_lip_z / std_lip_z, torch.tensor(float("nan"))
    )
    norm_lhand_z = torch.where(
        valid_mask[:, LHAND, -1], gap_lhand_z / std_lhand_z, torch.tensor(float("nan"))
    )
    norm_rhand_z = torch.where(
        valid_mask[:, RHAND, -1], gap_rhand_z / std_rhand_z, torch.tensor(float("nan"))
    )

    pose = norm_xy[:, LHAND + RHAND + LIP]
    pose_z = torch.cat((norm_lhand_z, norm_rhand_z, norm_lip_z), dim=-1)
    xyz = torch.cat((pose, pose_z.unsqueeze(-1)), dim=-1)

    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP]

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


def normalize_feature_3d(xyz):
    """
    Args:
        feat: [None, # landmark, 3]
    Return
        normalized_feat: [None, # landmark, 3]
    """
    valid_mask = ~torch.isnan(xyz)
    sum_xyz = torch.nansum(xyz, dim=(0, 1))
    count_xyz = torch.sum(valid_mask, dim=(0, 1)).to(torch.float32)
    mean_xyz = sum_xyz / count_xyz

    gap = torch.where(valid_mask, xyz - mean_xyz, torch.tensor(float("nan")))
    sum_sq_diff = torch.nansum(gap**2, dim=(0, 1))
    std_xyz = torch.sqrt(sum_sq_diff / count_xyz)

    xyz = torch.where(valid_mask, gap / std_xyz, torch.tensor(float("nan")))
    return xyz


def normalize_feature_1d(xyz):
    """
    Args:
        feat: [None]
    Return
        normalized_feat: [None]
    """
    valid_mask = ~torch.isnan(xyz)
    sum_xyz = torch.nansum(xyz)
    count_xyz = torch.sum(valid_mask).to(torch.float32)
    mean_xyz = sum_xyz / count_xyz

    gap = torch.where(valid_mask, xyz - mean_xyz, torch.tensor(float("nan")))
    sum_sq_diff = torch.nansum(gap**2)
    std_xyz = torch.sqrt(sum_sq_diff / count_xyz)

    xyz = torch.where(valid_mask, gap / std_xyz, torch.tensor(float("nan")))
    return xyz


def preprocess_xyzd_hdist_v6(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # global noramlization
    xyz = normalize_feature_3d(xyz)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # motion and normalize
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = normalize_feature_3d(dxyz)
    dxyz = np.pad(dxyz, [[0, 1], [0, 0], [0, 0]])

    # hand joint-wise distance and normalize
    mask = torch.tril(torch.ones(L, 21, 21, dtype=torch.bool), diagonal=-1)
    lhand = xyz[:, :21, :2]
    ld = lhand.reshape(-1, 21, 1, 2) - lhand.reshape(-1, 1, 21, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)
    ld = normalize_feature_1d(ld)

    rhand = xyz[:, 21:42, :2]
    rd = rhand.reshape(-1, 21, 1, 2) - rhand.reshape(-1, 1, 21, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)
    rd = normalize_feature_1d(rd)

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

    # interpolate (increase nan values)
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


def preprocess_xyzd_hdist_ma(xyz, max_len):
    pass


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


def preprocess_v2(xyz, max_len):
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    # lip_xy = xyz[:, LIP, :2]
    lhand_xy = xyz[:, LHAND, :2]
    rhand_xy = xyz[:, RHAND, :2]
    pose_xy = xyz[:, POSE, :2]

    # min-max scaling (part-wise)
    pose_min_x, pose_min_y, pose_max_x, pose_max_y = (
        torch.min(pose_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(pose_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(pose_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(pose_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    pose_min, pose_max = torch.cat((pose_min_x, pose_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((pose_max_x, pose_max_y), dim=-1).unsqueeze(1)
    pose_xy -= pose_min
    pose_xy /= pose_max - pose_min
    pose_center = (pose_max + pose_min) / 2

    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_xy -= lhand_min
    lhand_xy /= lhand_max - lhand_min
    lhand_center = (lhand_max + lhand_min) / 2

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_xy -= rhand_min
    rhand_xy /= rhand_max - rhand_min  # [None, 21, 2]
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]

    x = torch.cat(
        [
            pose_xy.reshape(L, -1),
            lhand_xy.reshape(L, -1),
            rhand_xy.reshape(L, -1),
            pose_center.reshape(L, -1),
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
        ],  # (none, 128)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


if __name__ == "__main__":
    import os
    from dataset import load_relevant_data_subset

    max_len = 384
    xyz = load_relevant_data_subset(
        "/sources/dataset/train_landmark_files/2044/635217.parquet"
    )
    xyz = torch.from_numpy(xyz).float()
    xyz = preprocess_v2(xyz, max_len)
    # print(xyz)
    print(xyz.shape)
    print(min(xyz[5]))
    print(max(xyz[5]))
