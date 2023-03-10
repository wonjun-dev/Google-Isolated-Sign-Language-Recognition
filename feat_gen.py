import numpy as np
from scipy.ndimage import zoom
import torch
import pandas as pd


class FeatureGen:
    def __init__(self, segments: int = 5):
        LEFT_HAND_OFFSET = 468
        POSE_OFFSET = LEFT_HAND_OFFSET + 21
        RIGHT_HAND_OFFSET = POSE_OFFSET + 33

        LIPSOUT_LM = [
            0,
            267,
            269,
            270,
            409,
            287,
            375,
            321,
            405,
            314,
            17,
            84,
            181,
            91,
            146,
            57,
            185,
            40,
            39,
            37,
        ]
        LIPSIN_LM = [
            13,
            312,
            311,
            310,
            415,
            308,
            324,
            318,
            402,
            317,
            14,
            87,
            178,
            88,
            95,
            78,
            191,
            80,
            81,
            82,
        ]

        PNOSE_LM = [0]
        PFACE_LM = [8, 6, 5, 4, 1, 2, 3, 7]
        BODY_LM = [11, 12, 24, 23]
        ARM_LM = [14, 16, 22, 20, 18, 13, 15, 21, 19, 17]

        lip_landmarks = LIPSIN_LM + LIPSOUT_LM
        pose_landmarks = PNOSE_LM + PFACE_LM + BODY_LM + ARM_LM
        left_hand_landmarks = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + 21))
        right_hand_landmarks = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + 21))

        self.point_landmarks = [
            item
            for sublist in [
                lip_landmarks,
                pose_landmarks,
                left_hand_landmarks,
                right_hand_landmarks,
            ]
            for item in sublist
        ]
        self.segments = segments

    def __call__(self, x):
        x = np.take(x, self.point_landmarks, axis=1)  # [N, 105, 3]
        n_frame, num_landmark, num_coord = x.shape[0], x.shape[1], x.shape[2]
        new_n_frame = n_frame + (self.segments - (n_frame % self.segments))
        x = zoom(
            x, (new_n_frame, num_landmark, num_coord) / np.array(x.shape), order=1
        )  # [N', num_ladmark, 3]

        x = torch.tensor(x, dtype=torch.float32)
        frame_per_seg = x.shape[0] // self.segments
        x = x.view(
            -1, frame_per_seg, 105, 3
        )  # [segments, frame_per_seg, num_landmark, 3]

        x_mean = fill_nan_zero(torch_nan_mean(x))  # [segments, num_landmark, 3]
        x_std = fill_nan_zero(torch_nan_std(x))  # [segments, num_landmark, 3]

        feat = torch.cat([x_mean, x_std], axis=0)  # [2*segments, num_landmark, 3]
        feat = feat.view(1, -1)  # [1, 2*segments * num_landmark * 3]

        return feat


ROWS_PER_FRAME = 543  # number of landmarks per frame


def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def torch_nan_mean(x, axis=1):
    nan_mask = torch.isnan(x)
    zero_mask = torch.zeros_like(x)
    ones_mask = torch.ones_like(x)

    # Replace NaN values with zeros
    x = torch.where(nan_mask, zero_mask, x)

    # Compute the sum of non-NaN values along the specified axis
    sum_values = torch.sum(x, dim=axis)
    count_values = torch.sum(torch.where(nan_mask, zero_mask, ones_mask), dim=axis)

    # Compute the mean
    mean_values = sum_values / count_values

    return mean_values


def torch_nan_std(x, axis=1):
    mean_values = torch_nan_mean(x, axis=axis)

    d = x - mean_values.unsqueeze(1)
    return torch.sqrt(torch_nan_mean(d * d, axis=axis))


def fill_nan_zero(x):
    nan_mask = torch.isnan(x)
    zero_mask = torch.zeros_like(x)

    # Replace NaN values with zeros
    x = torch.where(nan_mask, zero_mask, x)
    return x
