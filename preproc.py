import torch

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
