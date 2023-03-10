import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from model import ISLRModel, MLPLayer
from dataset import ISLRDataSet
from options import parser
from utils import AverageMeter, save_checkpoint, accuracy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def main():
    global args, best_acc
    args = parser.parse_args()

    ROOT_PATH = args.data_root_path

    if not os.path.exists(os.path.join(args.log_path, args.exp_name)):
        os.makedirs(os.path.join(args.log_path, args.exp_name))
    if not os.path.exists(os.path.join(args.ckpt_path, args.exp_name)):
        os.makedirs(os.path.join(args.ckpt_path, args.exp_name))

    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    for cur_fold in range(1, args.folds + 1):
        best_acc = 0

        print(f"####### Fold-{cur_fold} training start #######")
        os.makedirs(
            os.path.join(args.log_path, args.exp_name, time_stamp, str(cur_fold)),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(args.ckpt_path, args.exp_name, time_stamp, str(cur_fold)),
            exist_ok=True,
        )
        log_training = open(
            os.path.join(
                args.log_path, args.exp_name, time_stamp, str(cur_fold), "log.csv"
            ),
            "w",
        )
        tf_writer = SummaryWriter(
            os.path.join(args.log_path, args.exp_name, time_stamp, str(cur_fold))
        )

        train_idx = np.load(os.path.join(ROOT_PATH, "cv", f"train_idx_f{cur_fold}.npy"))
        val_idx = np.load(os.path.join(ROOT_PATH, "cv", f"val_idx_f{cur_fold}.npy"))

        train_dataset = ISLRDataSet(ver=args.data_ver, indicies=train_idx)
        val_dataset = ISLRDataSet(ver=args.data_ver, indicies=val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
        )

        ####### Model #######
        model = ISLRModel(encoder_layer=MLPLayer)
        try:
            model = nn.DataParallel(model).cuda()
        except:
            model = model.cuda()
        cudnn.benchmark = True

        ####### Optimizer #######
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        ####### Loss #######
        if args.loss == "ce":
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            raise NotImplementedError

        ####### Loop #######
        for epoch in range(1, args.epochs + 1):
            train(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                log_training,
                tf_writer,
            )

            if epoch % args.eval_freq == 0:
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
                    args.ckpt_path,
                    args.exp_name,
                    time_stamp,
                    cur_fold,
                    epoch,
                )
                tf_writer.close()


def train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer):
    losses = AverageMeter()
    accs = AverageMeter()

    model.train()
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        loss = criterion(output, y)
        acc = accuracy(output, y)

        losses.update(loss.item(), x.size(0))
        accs.update(acc, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx + 1) % args.print_freq == 0:
            info = (
                "Train: epoch-{0} ({1}/{2})\t"
                "loss {loss.avg:.5f}\t"
                "acc {acc.avg:.5f}\t".format(
                    epoch, idx + 1, len(train_loader), loss=losses, acc=accs
                )
            )
            losses.reset()
            accs.reset()
            print(info)
            log_training.write(info + "\n")
            log_training.flush()

    tf_writer.add_scalar("loss/train", losses.avg, epoch)
    tf_writer.add_scalar("acc/train", accs.avg, epoch)
    tf_writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)


def validate(val_loader, model, criterion, epoch, log_training, tf_writer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            acc = accuracy(output, y)

            losses.update(loss.item(), x.size(0))
            accs.update(acc, x.size(0))

    info = "Validate: Loss {loss.avg:.5f}\t Acc {acc.avg:.5f}".format(
        loss=losses, acc=accs
    )
    print(info)

    log_training.write(info + "\n")
    log_training.flush()
    tf_writer.add_scalar("loss/valid", losses.avg, epoch)
    tf_writer.add_scalar("acc/valid", accs.avg, epoch)

    return losses.avg, accs.avg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    seed_everything(seed=777)
    main()
