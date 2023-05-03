import torch
import numpy as np
from scipy.interpolate import interp1d
import torch.nn.functional as F

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


def arm_angle(xyz):
    p11, p13, p15, p23 = xyz[:, 85], xyz[:, 87], xyz[:, 89], xyz[:, 91]
    p12, p14, p16, p24 = xyz[:, 86], xyz[:, 88], xyz[:, 90], xyz[:, 92]
    v11_13, v11_23, v13_11, v13_15 = p13 - p11, p23 - p11, p11 - p13, p15 - p13
    v12_14, v12_24, v14_12, v14_16 = p14 - p12, p24 - p12, p12 - p14, p16 - p14
    angles = torch.stack(
        [
            get_angle(v11_13, v11_23),
            get_angle(v13_11, v13_15),
            get_angle(v12_14, v12_24),
            get_angle(v14_12, v14_16),
        ],
        dim=1,
    )

    return angles


def get_angle(v1, v2):
    nv1, nv2 = v1 / torch.norm(v1, dim=1, p=2, keepdim=True), v2 / torch.norm(
        v2, dim=1, p=2, keepdim=True
    )
    cos = torch.sum(nv1 * nv2, dim=1) / (
        (
            torch.sqrt(torch.sum(nv1**2, dim=1))
            * torch.sqrt(torch.sum(nv2**2, dim=1))
        )
        + 1e-10
    )

    return torch.acos(cos)


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
        feat: [None, # landmark, 3 or 2]
    Return
        normalized_feat: [None, # landmark, 3 or 2]
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
    """
    simplified 27 landmarks
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    # lip_xy = xyz[:, LIP, :2]
    overall_xy = xyz[:, POSE_SIM + LHAND_SIM + RHAND_SIM, :2]
    lhand_xy = xyz[:, LHAND_SIM, :2]
    rhand_xy = xyz[:, RHAND_SIM, :2]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_xy -= overall_min
    overall_xy /= overall_max - overall_min

    # 1-2. hand scaling
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
    rhand_xy /= rhand_max - rhand_min  # [None, 10, 2]

    # 2. location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]

    x = torch.cat(
        [
            overall_xy.reshape(L, -1),
            lhand_xy.reshape(L, -1),
            rhand_xy.reshape(L, -1),
            overall_center.reshape(L, -1),
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
        ],  # (none, 100)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v3(xyz, max_len):
    """
    v2 + lips
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    overall_xy = xyz[:, POSE_SIM + LHAND_SIM + RHAND_SIM + simple_lips, :2]
    lhand_xy = xyz[:, LHAND_SIM, :2]
    rhand_xy = xyz[:, RHAND_SIM, :2]
    lip_xy = xyz[:, simple_lips, :2]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_xy -= overall_min
    overall_xy /= overall_max - overall_min

    # 1-2. hand scaling
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
    rhand_xy /= rhand_max - rhand_min  # [None, 10, 2]

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_xy -= lip_min
    lip_xy /= lip_max - lip_min  # [None, 10, 2]

    # 2. location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    x = torch.cat(
        [
            overall_xy.reshape(L, -1),
            lhand_xy.reshape(L, -1),
            rhand_xy.reshape(L, -1),
            lip_xy.reshape(L, -1),
            overall_center.reshape(L, -1),
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
        ],  # (none, 134)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v4(xyz, max_len):
    """
    v3 + 손 temporal 변화 (속도)
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    overall_xy = xyz[:, POSE_SIM + LHAND_SIM + RHAND_SIM + simple_lips, :2]
    lhand_xy = xyz[:, LHAND_SIM, :2]
    rhand_xy = xyz[:, RHAND_SIM, :2]
    lip_xy = xyz[:, simple_lips, :2]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 10, 2]

    # 2. location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. 손 좌표 gobal 위치 temporal 변화 (속도)
    lhand_dxy = lhand_xy[:-1] - lhand_xy[1:]
    lhand_dxy = torch.from_numpy(np.pad(lhand_dxy, [[0, 1], [0, 0], [0, 0]]))
    rhand_dxy = rhand_xy[:-1] - rhand_xy[1:]
    rhand_dxy = torch.from_numpy(np.pad(rhand_dxy, [[0, 1], [0, 0], [0, 0]]))

    x = torch.cat(
        [
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            lhand_dxy.reshape(L, -1),  # velocity of hand landmarks
            rhand_dxy.reshape(L, -1),
        ],  # (none, 174)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v5(xyz, max_len):
    """
    v4 + 손 관절 사이의 거리 (hdist)
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    overall_xy = xyz[:, POSE_SIM + LHAND_SIM + RHAND_SIM + simple_lips, :2]
    lhand_xy = xyz[:, LHAND_SIM, :2]
    rhand_xy = xyz[:, RHAND_SIM, :2]
    lip_xy = xyz[:, simple_lips, :2]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 10, 2]

    # 2. location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. 손 좌표 gobal 위치 temporal 변화 (속도)
    lhand_dxy = lhand_xy[:-1] - lhand_xy[1:]
    lhand_dxy = torch.from_numpy(np.pad(lhand_dxy, [[0, 1], [0, 0], [0, 0]]))
    rhand_dxy = rhand_xy[:-1] - rhand_xy[1:]
    rhand_dxy = torch.from_numpy(np.pad(rhand_dxy, [[0, 1], [0, 0], [0, 0]]))

    # 4. 손 scaled 좌표 사이의 거리
    mask = torch.tril(torch.ones(L, 10, 10, dtype=torch.bool), diagonal=-1)
    ld = lhand_scaled_xy.reshape(-1, 10, 1, 2) - lhand_scaled_xy.reshape(-1, 1, 10, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rd = rhand_scaled_xy.reshape(-1, 10, 1, 2) - rhand_scaled_xy.reshape(-1, 1, 10, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            lhand_dxy.reshape(L, -1),  # velocity of global hand landmarks
            rhand_dxy.reshape(L, -1),
            ld.reshape(L, -1),  # distance between scalsed hand landmarks
            rd.reshape(L, -1),
        ],  # (none, 264)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v6(xyz, max_len):
    """
    x, y 정규화 + v5
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    overall_xy = xyz[:, LHAND_SIM + RHAND_SIM + simple_lips + POSE_SIM, :2]
    overall_xy = normalize_feature_3d(overall_xy)

    lhand_xy = overall_xy[:, : len(LHAND_SIM), :2]
    rhand_xy = overall_xy[:, len(LHAND_SIM) : len(LHAND_SIM) + len(RHAND_SIM), :2]
    lip_xy = overall_xy[
        :,
        len(LHAND_SIM)
        + len(RHAND_SIM) : len(LHAND_SIM)
        + len(RHAND_SIM)
        + len(simple_lips),
        :2,
    ]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min
    overall_scaled_xy -= 0.5

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min
    lhand_scaled_xy -= 0.5

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]
    rhand_scaled_xy -= 0.5

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 8, 2]
    lip_scaled_xy -= 0.5

    # 2. global location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. 손 좌표 gobal 위치 temporal 변화 (속도)
    lhand_dxy = lhand_xy[:-1] - lhand_xy[1:]
    lhand_dxy = torch.from_numpy(np.pad(lhand_dxy, [[0, 1], [0, 0], [0, 0]]))
    rhand_dxy = rhand_xy[:-1] - rhand_xy[1:]
    rhand_dxy = torch.from_numpy(np.pad(rhand_dxy, [[0, 1], [0, 0], [0, 0]]))

    # 4. 손 scaled 좌표 사이의 거리
    mask = torch.tril(torch.ones(L, 10, 10, dtype=torch.bool), diagonal=-1)
    ld = lhand_scaled_xy.reshape(-1, 10, 1, 2) - lhand_scaled_xy.reshape(-1, 1, 10, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rd = rhand_scaled_xy.reshape(-1, 10, 1, 2) - rhand_scaled_xy.reshape(-1, 1, 10, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            lhand_dxy.reshape(L, -1),  # velocity of global hand landmarks
            rhand_dxy.reshape(L, -1),
            ld.reshape(L, -1),  # distance between scalsed hand landmarks
            rd.reshape(L, -1),
        ],  # (none, 264)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v7(xyz, max_len):
    """
    v6 + normalized overall, hand velocity -> overall velocity
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    overall_xy = xyz[:, LHAND_SIM + RHAND_SIM + simple_lips + POSE_SIM, :2]
    overall_xy = normalize_feature_3d(overall_xy)

    lhand_xy = overall_xy[:, : len(LHAND_SIM), :2]
    rhand_xy = overall_xy[:, len(LHAND_SIM) : len(LHAND_SIM) + len(RHAND_SIM), :2]
    lip_xy = overall_xy[
        :,
        len(LHAND_SIM)
        + len(RHAND_SIM) : len(LHAND_SIM)
        + len(RHAND_SIM)
        + len(simple_lips),
        :2,
    ]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min
    overall_scaled_xy -= 0.5

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min
    lhand_scaled_xy -= 0.5

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]
    rhand_scaled_xy -= 0.5

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 8, 2]
    lip_scaled_xy -= 0.5

    # 2. global location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. 좌표 global 위치 temporal 변화 (속도)
    overall_dxy = overall_xy[:-1] - overall_xy[1:]
    overall_dxy = torch.from_numpy(np.pad(overall_dxy, [[0, 1], [0, 0], [0, 0]]))

    # 4. 손 scaled 좌표 사이의 거리
    mask = torch.tril(torch.ones(L, 10, 10, dtype=torch.bool), diagonal=-1)
    ld = lhand_scaled_xy.reshape(-1, 10, 1, 2) - lhand_scaled_xy.reshape(-1, 1, 10, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rd = rhand_scaled_xy.reshape(-1, 10, 1, 2) - rhand_scaled_xy.reshape(-1, 1, 10, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            overall_xy.reshape(L, -1),  # normalized landmark
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            overall_dxy.reshape(L, -1),  # velocity of global landmarks
            ld.reshape(L, -1),  # distance between scalsed hand landmarks
            rd.reshape(L, -1),
        ],  # (none, 364)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v8(xyz, max_len):
    """
    v7 + ld, rd -> global 기준
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    overall_xy = xyz[:, LHAND_SIM + RHAND_SIM + simple_lips + POSE_SIM, :2]
    overall_xy = normalize_feature_3d(overall_xy)

    lhand_xy = overall_xy[:, : len(LHAND_SIM), :2]
    rhand_xy = overall_xy[:, len(LHAND_SIM) : len(LHAND_SIM) + len(RHAND_SIM), :2]
    lip_xy = overall_xy[
        :,
        len(LHAND_SIM)
        + len(RHAND_SIM) : len(LHAND_SIM)
        + len(RHAND_SIM)
        + len(simple_lips),
        :2,
    ]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min
    overall_scaled_xy -= 0.5

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min
    lhand_scaled_xy -= 0.5

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]
    rhand_scaled_xy -= 0.5

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 8, 2]
    lip_scaled_xy -= 0.5

    # 2. global location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. 좌표 global 위치 temporal 변화 (속도)
    overall_dxy = overall_xy[:-1] - overall_xy[1:]
    overall_dxy = torch.from_numpy(np.pad(overall_dxy, [[0, 1], [0, 0], [0, 0]]))

    # 4. 손 global 좌표 사이의 거리
    mask = torch.tril(torch.ones(L, 10, 10, dtype=torch.bool), diagonal=-1)
    ld = lhand_xy.reshape(-1, 10, 1, 2) - lhand_xy.reshape(-1, 1, 10, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rd = rhand_xy.reshape(-1, 10, 1, 2) - rhand_xy.reshape(-1, 1, 10, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            overall_xy.reshape(L, -1),  # normalized landmark
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            overall_dxy.reshape(L, -1),  # velocity of global landmarks
            ld.reshape(L, -1),  # distance between hand global landmarks
            rd.reshape(L, -1),
        ],  # (none, 364)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v9(xyz, max_len):
    """
    v9 + bbox 들간의 거리
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    overall_xy = xyz[:, LHAND_SIM + RHAND_SIM + simple_lips + POSE_SIM, :2]
    overall_xy = normalize_feature_3d(overall_xy)

    lhand_xy = overall_xy[:, : len(LHAND_SIM), :2]
    rhand_xy = overall_xy[:, len(LHAND_SIM) : len(LHAND_SIM) + len(RHAND_SIM), :2]
    lip_xy = overall_xy[
        :,
        len(LHAND_SIM)
        + len(RHAND_SIM) : len(LHAND_SIM)
        + len(RHAND_SIM)
        + len(simple_lips),
        :2,
    ]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min
    overall_scaled_xy -= 0.5

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min
    lhand_scaled_xy -= 0.5

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]
    rhand_scaled_xy -= 0.5

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 8, 2]
    lip_scaled_xy -= 0.5

    # 2. global location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. bbox 들간의 거리
    mask = torch.tril(torch.ones(L, 4, 4, dtype=torch.bool), diagonal=-1)
    bbox_xy = torch.cat((overall_center, lhand_center, rhand_center, lip_center), dim=1)
    bboxd = bbox_xy.reshape(-1, 4, 1, 2) - bbox_xy.reshape(-1, 1, 4, 2)
    bboxd = torch.sqrt((bboxd**2).sum(-1))
    bboxd = bboxd.masked_select(mask)  # [None * 6]

    # 4. 좌표 global 위치 temporal 변화 (속도)
    overall_dxy = overall_xy[:-1] - overall_xy[1:]
    overall_dxy = torch.from_numpy(np.pad(overall_dxy, [[0, 1], [0, 0], [0, 0]]))

    # 5. 손 global 좌표 사이의 거리
    mask = torch.tril(torch.ones(L, 10, 10, dtype=torch.bool), diagonal=-1)
    ld = lhand_xy.reshape(-1, 10, 1, 2) - lhand_xy.reshape(-1, 1, 10, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rd = rhand_xy.reshape(-1, 10, 1, 2) - rhand_xy.reshape(-1, 1, 10, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            overall_xy.reshape(L, -1),  # normalized landmark
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            bboxd.reshape(L, -1),  # distance between bbox global landmarks
            overall_dxy.reshape(L, -1),  # velocity of global landmarks
            ld.reshape(L, -1),  # distance between hand global landmarks
            rd.reshape(L, -1),
        ],  # (none, 364)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v1(xyz, max_len):
    """
    xyzd_hdist_v2 에서 selection 먼저
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # noramlization
    xyz = normalize_feature_3d(xyz)

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


def preprocess_v1_1(xyz, max_len):
    """
    v1 + no z
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # selection
    xyz = xyz[:, LHAND + RHAND + LIP, :2]

    # noramlization
    xyz = normalize_feature_3d(xyz)

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


def preprocess_v8_1(xyz, max_len):
    """
    v8 + selection, normalize -> normalize, selection
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlize
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    overall_xy = xyz[:, LHAND_SIM + RHAND_SIM + simple_lips + POSE_SIM, :2]

    lhand_xy = overall_xy[:, : len(LHAND_SIM), :2]
    rhand_xy = overall_xy[:, len(LHAND_SIM) : len(LHAND_SIM) + len(RHAND_SIM), :2]
    lip_xy = overall_xy[
        :,
        len(LHAND_SIM)
        + len(RHAND_SIM) : len(LHAND_SIM)
        + len(RHAND_SIM)
        + len(simple_lips),
        :2,
    ]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min
    overall_scaled_xy -= 0.5

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min
    lhand_scaled_xy -= 0.5

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]
    rhand_scaled_xy -= 0.5

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 8, 2]
    lip_scaled_xy -= 0.5

    # 2. global location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. 좌표 global 위치 temporal 변화 (속도)
    overall_dxy = overall_xy[:-1] - overall_xy[1:]
    overall_dxy = torch.from_numpy(np.pad(overall_dxy, [[0, 1], [0, 0], [0, 0]]))

    # 4. 손 global 좌표 사이의 거리
    mask = torch.tril(torch.ones(L, 10, 10, dtype=torch.bool), diagonal=-1)
    ld = lhand_xy.reshape(-1, 10, 1, 2) - lhand_xy.reshape(-1, 1, 10, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rd = rhand_xy.reshape(-1, 10, 1, 2) - rhand_xy.reshape(-1, 1, 10, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            overall_xy.reshape(L, -1),  # normalized landmark
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            overall_dxy.reshape(L, -1),  # velocity of global landmarks
            ld.reshape(L, -1),  # distance between hand global landmarks
            rd.reshape(L, -1),
        ],  # (none, 364)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preprocess_v9_1(xyz, max_len):
    """
    v9 + selection, normalize -> normalize, selection
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlize
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    overall_xy = xyz[:, LHAND_SIM + RHAND_SIM + simple_lips + POSE_SIM, :2]

    lhand_xy = overall_xy[:, : len(LHAND_SIM), :2]
    rhand_xy = overall_xy[:, len(LHAND_SIM) : len(LHAND_SIM) + len(RHAND_SIM), :2]
    lip_xy = overall_xy[
        :,
        len(LHAND_SIM)
        + len(RHAND_SIM) : len(LHAND_SIM)
        + len(RHAND_SIM)
        + len(simple_lips),
        :2,
    ]

    # 1. min-max scaling (part-wise)
    # 1-1. overall scaling
    overall_min_x, overall_min_y, overall_max_x, overall_max_y = (
        torch.min(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(overall_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(overall_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    overall_min, overall_max = torch.cat(
        (overall_min_x, overall_min_y), dim=-1
    ).unsqueeze(1), torch.cat((overall_max_x, overall_max_y), dim=-1).unsqueeze(1)
    overall_scaled_xy = overall_xy - overall_min
    overall_scaled_xy /= overall_max - overall_min
    overall_scaled_xy -= 0.5

    # 1-2. hand scaling
    lhand_min_x, lhand_min_y, lhand_max_x, lhand_max_y = (
        torch.min(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lhand_min, lhand_max = torch.cat((lhand_min_x, lhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lhand_max_x, lhand_max_y), dim=-1).unsqueeze(1)
    lhand_scaled_xy = lhand_xy - lhand_min
    lhand_scaled_xy /= lhand_max - lhand_min
    lhand_scaled_xy -= 0.5

    rhand_min_x, rhand_min_y, rhand_max_x, rhand_max_y = (
        torch.min(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(rhand_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    rhand_min, rhand_max = torch.cat((rhand_min_x, rhand_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((rhand_max_x, rhand_max_y), dim=-1).unsqueeze(1)
    rhand_scaled_xy = rhand_xy - rhand_min
    rhand_scaled_xy /= rhand_max - rhand_min  # [None, 10, 2]
    rhand_scaled_xy -= 0.5

    lip_min_x, lip_min_y, lip_max_x, lip_max_y = (
        torch.min(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.min(lip_xy[:, :, 1], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 0], dim=1, keepdims=True).values,
        torch.max(lip_xy[:, :, 1], dim=1, keepdims=True).values,
    )
    lip_min, lip_max = torch.cat((lip_min_x, lip_min_y), dim=-1).unsqueeze(
        1
    ), torch.cat((lip_max_x, lip_max_y), dim=-1).unsqueeze(1)
    lip_scaled_xy = lip_xy - lip_min
    lip_scaled_xy /= lip_max - lip_min  # [None, 8, 2]
    lip_scaled_xy -= 0.5

    # 2. global location
    overall_center = (overall_max + overall_min) / 2
    lhand_center = (lhand_max + lhand_min) / 2
    rhand_center = (rhand_max + rhand_min) / 2  # [None, 1, 2]
    lip_center = (lip_max + lip_min) / 2

    # 3. bbox 들간의 거리
    mask = torch.tril(torch.ones(L, 4, 4, dtype=torch.bool), diagonal=-1)
    bbox_xy = torch.cat((overall_center, lhand_center, rhand_center, lip_center), dim=1)
    bboxd = bbox_xy.reshape(-1, 4, 1, 2) - bbox_xy.reshape(-1, 1, 4, 2)
    bboxd = torch.sqrt((bboxd**2).sum(-1))
    bboxd = bboxd.masked_select(mask)  # [None * 6]

    # 4. 좌표 global 위치 temporal 변화 (속도)
    overall_dxy = overall_xy[:-1] - overall_xy[1:]
    overall_dxy = torch.from_numpy(np.pad(overall_dxy, [[0, 1], [0, 0], [0, 0]]))

    # 5. 손 global 좌표 사이의 거리
    mask = torch.tril(torch.ones(L, 10, 10, dtype=torch.bool), diagonal=-1)
    ld = lhand_xy.reshape(-1, 10, 1, 2) - lhand_xy.reshape(-1, 1, 10, 2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.masked_select(mask)

    rd = rhand_xy.reshape(-1, 10, 1, 2) - rhand_xy.reshape(-1, 1, 10, 2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.masked_select(mask)

    x = torch.cat(
        [
            overall_xy.reshape(L, -1),  # normalized landmark
            overall_scaled_xy.reshape(L, -1),  # scaled position
            lhand_scaled_xy.reshape(L, -1),
            rhand_scaled_xy.reshape(L, -1),
            lip_scaled_xy.reshape(L, -1),
            overall_center.reshape(L, -1),  # global location of bbox
            lhand_center.reshape(L, -1),
            rhand_center.reshape(L, -1),
            lip_center.reshape(L, -1),
            bboxd.reshape(L, -1),  # distance between bbox global landmarks
            overall_dxy.reshape(L, -1),  # velocity of global landmarks
            ld.reshape(L, -1),  # distance between hand global landmarks
            rd.reshape(L, -1),
        ],  # (none, 364)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0(xyz, max_len):
    """
    same as xyzd_hdist_v2
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz)
    # selection
    xyz = xyz[:, LHAND + RHAND + LIP]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 912)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_1(xyz, max_len):
    """
    v0 + no z
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    xyz = xyz[:, LHAND + RHAND + LIP, :2]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 912)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_2(xyz, max_len):
    """
    v0 + no lips
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz)
    # selection
    xyz = xyz[:, LHAND + RHAND]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 912)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_3(xyz, max_len):
    """
    v0 + lips -> simple lips(v03, pointdim 720) or SLIP(v031, pointdim 792)
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz)
    # selection
    # xyz = xyz[:, LHAND + RHAND + simple_lips]
    xyz = xyz[:, LHAND + RHAND + SLIP]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 912)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_4(xyz, max_len):
    """
    v0 + simple pose
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz)
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 966, 978)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_5(xyz, max_len):
    """
    v0 + v0_1 + v0_3 (SLIP) + v0_4(966)
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    xyz = xyz[:, LHAND + RHAND + SLIP + POSE_SIM, :2]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 704)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_6(xyz, max_len):
    """
    v0 + v0_1 + v0_4(966)
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE, :2]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 784 or 792)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_7(xyz, max_len):
    """
    v0_6 + leye, reye, nose
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM + LEYE + REYE + NOSE, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE + LEYE + REYE + NOSE, :2]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
        ],  # (none, 928 or)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_8(xyz, max_len):
    """
    v0_7 + some joint distances
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM + LEYE + REYE + NOSE, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE + LEYE + REYE + NOSE, :2]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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

    # pose joint-wise distance
    mask = torch.tril(torch.ones(L, 11, 11, dtype=torch.bool), diagonal=-1)
    spose = xyz[:, 82:93, :2]
    pd = spose.reshape(-1, 11, 1, 2) - spose.reshape(-1, 1, 11, 2)
    pd = torch.sqrt((pd**2).sum(-1))
    pd = pd.masked_select(mask)

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            pd.reshape(L, -1),
        ],  # (none, 991)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_9(xyz, max_len):
    """
    v0_7 + reverse 프레임 차이
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM + LEYE + REYE + NOSE, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE + LEYE + REYE + NOSE, :2]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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

    # reverse difference
    rdxyz = xyz - xyz.flip(dims=[0])

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            rdxyz.reshape(L, -1),
        ],  # (none, 1194)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_91(xyz, max_len=256, window=64):
    """
    v0_9 + fixed frame size (interpolation)
    """
    L = len(xyz)
    if L > max_len:
        i = (L - max_len) // 2
        xyz = xyz[i : i + max_len]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM + LEYE + REYE + NOSE, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE + LEYE + REYE + NOSE, :2]

    # interpolation
    L, V, C = xyz.shape
    xyz = xyz.permute(1, 2, 0).contiguous().view(V * C, L)
    xyz = xyz[None, None, :, :]
    xyz = F.interpolate(
        xyz, size=(V * C, window), mode="bilinear", align_corners=False
    ).squeeze()
    xyz = xyz.view(V, C, -1).permute(2, 0, 1).contiguous()  # [L, V, C]
    L, V, C = xyz.shape

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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

    # reverse difference
    rdxyz = xyz - xyz.flip(dims=[0])

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            rdxyz.reshape(L, -1),
        ],  # (none, 1194)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_92(xyz, max_len=384, window=64):
    """
    v0_91 + 1. crop center -> smapling, 2. interpolation dynamically
    """
    L = len(xyz)
    if L > max_len:
        step = (L - 1) // (max_len - 1)
        indices = [i * step for i in range(max_len)]
        xyz = xyz[indices]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM + LEYE + REYE + NOSE, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE + LEYE + REYE + NOSE, :2]

    # interpolation
    L, V, C = xyz.shape
    xyz = xyz.permute(1, 2, 0).contiguous().view(V * C, L)
    xyz = xyz[None, None, :, :]

    if L <= window:
        resize = min(2 * L, window)
    elif L > window:
        resize = min(L // 2, window)

    xyz = F.interpolate(
        xyz, size=(V * C, resize), mode="bilinear", align_corners=False
    ).squeeze()
    xyz = xyz.view(V, C, -1).permute(2, 0, 1).contiguous()  # [L, V, C]
    L, V, C = xyz.shape

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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

    # reverse difference
    rdxyz = xyz - xyz.flip(dims=[0])

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            rdxyz.reshape(L, -1),
        ],  # (none, 1194)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_93(xyz, max_len):
    """
    v0_9 + 1. center crop -> sampling
    """
    L = len(xyz)
    if L > max_len:
        step = (L - 1) // (max_len - 1)
        indices = [i * step for i in range(max_len)]
        xyz = xyz[indices]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM + LEYE + REYE + NOSE, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE + LEYE + REYE + NOSE, :2]

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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

    # reverse difference
    rdxyz = xyz - xyz.flip(dims=[0])

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            rdxyz.reshape(L, -1),
        ],  # (none, 1194)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


def preproc_v0_94(xyz, max_len):
    """
    v0_9 + 1. center crop -> sampling, 2. angle
    """
    L = len(xyz)
    if L > max_len:
        step = (L - 1) // (max_len - 1)
        indices = [i * step for i in range(max_len)]
        xyz = xyz[indices]
    L = len(xyz)

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    # xyz = xyz[:, LHAND + RHAND + LIP + POSE_SIM + LEYE + REYE + NOSE, :2]
    xyz = xyz[:, LHAND + RHAND + LIP + SPOSE + LEYE + REYE + NOSE, :2]

    ## angles
    angles = arm_angle(xyz)

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

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

    # reverse difference
    rdxyz = xyz - xyz.flip(dims=[0])

    x = torch.cat(
        [
            xyz.reshape(L, -1),
            dxyz.reshape(L, -1),
            ld.reshape(L, -1),
            rd.reshape(L, -1),
            rdxyz.reshape(L, -1),
            angles.reshape(L, -1),
        ],  # (none, 1198)
        -1,
    )
    x[torch.isnan(x)] = 0
    return x


# HyperFormer
def preproc_v1(xyz, max_len=256, window=64):
    # resize
    T, V, C = xyz.shape
    if T > max_len:
        i = (T - max_len) // 2
        xyz = xyz[i : i + max_len]
    T = xyz.shape[0]

    xyz = xyz.permute(1, 2, 0).contiguous().view(V * C, T)
    xyz = xyz[None, None, :, :]
    xyz = F.interpolate(
        xyz, size=(V * C, window), mode="bilinear", align_corners=False
    ).squeeze()
    xyz = xyz.view(V, C, -1).permute(1, 2, 0).contiguous()  # C, T, V

    # select
    xyz = xyz[:2, :, LHAND + RHAND + SPOSE]
    mask = torch.zeros_like(xyz)
    mask[torch.isnan(xyz)] = 1
    xyz[torch.isnan(xyz)] = 0
    return xyz, mask[0]


# GCN embedding
def preproc_v1_1(xyz, max_len=384):
    # resize
    T, V, C = xyz.shape
    if T > max_len:
        i = (T - max_len) // 2
        xyz = xyz[i : i + max_len]
    T = xyz.shape[0]

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    xyz = xyz[:, LHAND + RHAND + SPOSE, :2]  # [T, 53, 2]
    xyz[torch.isnan(xyz)] = 0
    return xyz


# v1-1 + dxy
def preproc_v1_2(xyz, max_len=384):
    # resize
    T, V, C = xyz.shape
    if T > max_len:
        i = (T - max_len) // 2
        xyz = xyz[i : i + max_len]
    T = xyz.shape[0]

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    xyz = xyz[:, LHAND + RHAND + SPOSE, :2]  # [T, 53, 2]
    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

    xyz = torch.cat(
        [xyz, dxyz],  # (T, 53, 4)
        -1,
    )

    xyz[torch.isnan(xyz)] = 0
    return xyz


# v1_2 + distance
def preproc_v1_3(xyz, max_len=384, window=64):
    # resize
    T, V, C = xyz.shape
    if T > max_len:
        step = (T - 1) // (max_len - 1)
        indices = [i * step for i in range(max_len)]
        xyz = xyz[indices]
    T = xyz.shape[0]

    # noramlization
    xyz = normalize_feature_3d(xyz[:, :, :2])
    # selection
    xyz = xyz[:, LHAND + RHAND + SPOSE, :2]  # [T, 53, 2]

    # interpolation
    T, V, C = xyz.shape
    xyz = xyz.permute(1, 2, 0).contiguous().view(V * C, T)
    xyz = xyz[None, None, :, :]

    if T <= window:
        resize = min(2 * T, window)
    elif T > window:
        resize = min(T // 2, window)

    xyz = F.interpolate(
        xyz, size=(V * C, resize), mode="bilinear", align_corners=False
    ).squeeze()
    xyz = xyz.view(V, C, -1).permute(2, 0, 1).contiguous()  # [T, V, C]
    T, V, C = xyz.shape

    # motion
    dxyz = xyz[:-1] - xyz[1:]
    dxyz = torch.from_numpy(np.pad(dxyz, [[0, 1], [0, 0], [0, 0]]))

    # joint-wise distance
    mask = (1 - torch.eye(53)).to(torch.bool)
    mask = mask.repeat(T, 1, 1)
    distance = xyz.reshape(-1, 53, 1, 2) - xyz.reshape(-1, 1, 53, 2)
    distance = torch.sqrt((distance**2).sum(-1))
    distance = distance.masked_select(mask)
    distance = distance.reshape(T, 53, 52)

    # reverse difference
    rdxyz = xyz - xyz.flip(dims=[0])

    xyz = torch.cat(
        [xyz, dxyz, distance, rdxyz],  # (T, 53, 58)
        -1,
    )

    xyz[torch.isnan(xyz)] = 0
    return xyz


if __name__ == "__main__":
    import os
    from dataset import load_relevant_data_subset

    max_len = 384
    xyz = load_relevant_data_subset(
        "/sources/dataset/train_landmark_files/2044/635217.parquet"
    )
    xyz = torch.from_numpy(xyz).float()
    xyz = preproc_v0_93(xyz, max_len)
    print(xyz.shape)
    print(xyz)
    # print(mask.shape)
    # print(xyz[5, 124:142])
    # print(xyz.shape)
    # for i in range(64):
    #     print(raw_xyz[i][:, 0])
    #     print(xyz[0][i])
    #     print('---')

    #     if i == 5:
    #         break
    print(min(xyz[5]))
    print(max(xyz[5]))
