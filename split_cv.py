import os
import random
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# check distribution
def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f"{y_distr[i]/y_vals_sum:.2%}" for i in range(np.max(y).astype(int) + 1)]


if __name__ == "__main__":
    seed_everything(seed=777)
    ROOT_PATH = "/sources/dataset/"
    os.makedirs(os.path.join(ROOT_PATH, "cv"), exist_ok=True)

    features = np.load(os.path.join(ROOT_PATH, "features", "ver0", "feature_data.npy"))
    labels = np.load(os.path.join(ROOT_PATH, "features", "ver0", "feature_labels.npy"))
    groups = np.load(os.path.join(ROOT_PATH, "participants.npy"))
    label_map = json.load(
        open(os.path.join(ROOT_PATH, "sign_to_prediction_index_map.json"))
    )

    cv = StratifiedGroupKFold(n_splits=5)

    distrs = [get_distribution(labels)]
    index = ["training set"]

    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(features, labels, groups)):
        train_y, val_y = labels[train_idx], labels[val_idx]
        train_gr, val_gr = groups[train_idx], groups[val_idx]

        assert len(set(train_gr) & set(val_gr)) == 0

        distrs.append(get_distribution(train_y))
        distrs.append(get_distribution(val_y))

        index.append(f"train - fold{fold_ind}")
        index.append(f"val - fold{fold_ind}")

        np.save(
            os.path.join(ROOT_PATH, "cv", f"train_idx_f{fold_ind+1}.npy"), train_idx
        )
        np.save(os.path.join(ROOT_PATH, "cv", f"val_idx_f{fold_ind+1}.npy"), val_idx)

    categories = [label for label in label_map.keys()]

    stats = pd.DataFrame(
        distrs,
        index=index,
        columns=[categories[i] for i in range(np.max(labels).astype(int) + 1)],
    )

    stats.to_csv(os.path.join(ROOT_PATH, "cv", "cv_stats.csv"))
