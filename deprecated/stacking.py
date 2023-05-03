import os
import random
import pandas as pd

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter, accuracy, save_checkpoint
from datetime import datetime
from preproc import preproc_v0_93

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

ver = "arcface"
exp_name = f"stacking/{ver}"
cur_fold = 1


class StackDataset(Dataset):
    def __init__(
        self, path="/sources/dataset/stack", ver="arcface", folds=[1, 2, 3, 4]
    ):
        data_path = os.path.join(path, ver)

        features = []
        labels = []
        indices = []
        for f in folds:
            features.append(np.load(os.path.join(data_path, f"X_fold{f}.npy")))
            labels.append(np.load(os.path.join(data_path, f"y_fold{f}.npy")))
            indices.append(
                np.load(os.path.join("/sources/dataset/cv", f"train_idx_f{f}.npy"))
            )

        self.features = np.concatenate(features, axis=0)
        self.labels = np.concatenate(labels, axis=0)
        self.indices = np.concatenate(indices, axis=0)

        df = pd.read_csv(os.path.join("/sources/dataset", "train.csv"))
        self.df = df.iloc[self.indices]
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        xyz = load_relevant_data_subset(
            os.path.join("/sources/dataset", self.df.iloc[index].path)
        )
        xyz = torch.from_numpy(xyz)
        xyz = preproc_v0_93(xyz, 64)
        xyz = torch.mean(xyz, dim=0)
        preds = torch.tensor(self.features[index, :], dtype=torch.float32)

        feat = torch.concat((xyz, preds), dim=0)
        return feat, self.labels[index].astype(int)


ROWS_PER_FRAME = 543


def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.l1 = nn.Linear(1444, 250, bias=False)
        self.bn = nn.BatchNorm1d(250)
        self.act = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(250, 250)

    def forward(self, x):
        logit = self.l2(self.act(self.bn(self.l1(self.norm(x)))))
        logit = self.l1(x)
        return logit


def main():
    global best_acc
    best_acc = 0

    if not os.path.exists(os.path.join("/sources/log", exp_name)):
        os.makedirs(os.path.join("/sources/log", exp_name))
    if not os.path.exists(os.path.join("/sources/ckpts", exp_name)):
        os.makedirs(os.path.join("/sources/ckpts", exp_name))

    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    os.makedirs(
        os.path.join("/sources/log", exp_name, time_stamp, str(cur_fold)),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join("/sources/ckpts", exp_name, time_stamp, str(cur_fold)),
        exist_ok=True,
    )
    log_training = open(
        os.path.join("/sources/log", exp_name, time_stamp, str(cur_fold), "log.csv"),
        "w",
    )
    tf_writer = SummaryWriter(
        os.path.join("/sources/log", exp_name, time_stamp, str(cur_fold))
    )

    train_dataset = StackDataset(
        ver=ver, folds=[i for i in range(1, 5) if i != cur_fold]
    )
    val_dataset = StackDataset(ver=ver, folds=[cur_fold])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )

    model = MetaLearner()
    model.cuda()

    ##### Config #####
    epochs = 10
    lb = 0.1
    lr = 1e-4
    criterion = nn.CrossEntropyLoss(label_smoothing=lb).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, log_training, tf_writer
        )
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        output_best = "Valid Best Acc: %.5f\n" % (best_acc)
        print(output_best)
        log_training.write(output_best + "\n")
        log_training.flush()

        save_checkpoint(
            {"state_dict": model.state_dict()},
            is_best,
            "/sources/ckpts",
            exp_name,
            time_stamp,
            cur_fold,
            epoch,
        )
        tf_writer.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    log_training,
    tf_writer,
):
    losses = AverageMeter()
    accs = AverageMeter()

    model.train()
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        output = model(x)

        loss = criterion(output, y)
        acc = accuracy(output, y)

        losses.update(loss.item(), y.size(0))
        accs.update(acc, y.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        info = (
            "Train: epoch-{0} ({1}/{2})\t"
            "loss {loss.avg:.5f}\t"
            "acc {acc.avg:.5f}\t".format(
                epoch, idx + 1, len(train_loader), loss=losses, acc=accs
            )
        )
        print(info)
        log_training.write(info + "\n")
        log_training.flush()

    tf_writer.add_scalar("loss/train", losses.avg, epoch)
    tf_writer.add_scalar("acc/train", accs.avg, epoch)
    tf_writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)

    losses.reset()
    accs.reset()


def validate(val_loader, model, criterion, epoch, log_training, tf_writer, margin=None):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x, y = x.cuda(), y.cuda()
            output = model(x)

            loss = criterion(output, y)
            acc = accuracy(output, y)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

    info = "Validate: Loss {loss.avg:.5f}\t Acc {acc.avg:.5f}".format(
        loss=losses, acc=accs
    )
    print(info)

    log_training.write(info + "\n")
    log_training.flush()
    tf_writer.add_scalar("loss/valid", losses.avg, epoch)
    tf_writer.add_scalar("acc/valid", accs.avg, epoch)

    return losses.avg, accs.avg


if __name__ == "__main__":
    seed_everything(seed=777)
    main()
