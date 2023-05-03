import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.backends.cudnn as cudnn
import random
import numpy as np
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from model import (
    ISLRModelV6,
    ISLRModelArcFace,
    ISLRModelArcFaceCE,
)
from dataset import ISLRDataSetV2, collate_func
from options import parser
from utils import AverageMeter, save_checkpoint, accuracy
from arcface import ArcMarginProduct

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    global args, best_acc
    args = parser.parse_args()

    ROOT_PATH = args.data_root_path
    SEED = [999, 777, 555, 333]

    if not os.path.exists(os.path.join(args.log_path, args.exp_name)):
        os.makedirs(os.path.join(args.log_path, args.exp_name))
    if not os.path.exists(os.path.join(args.ckpt_path, args.exp_name)):
        os.makedirs(os.path.join(args.ckpt_path, args.exp_name))

    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    for cur_seed in SEED:
        best_acc = 0

        seed_everything(seed=cur_seed)
        print(f"####### Seed-{cur_seed} training start #######")
        os.makedirs(
            os.path.join(args.log_path, args.exp_name, time_stamp, str(cur_seed)),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(args.ckpt_path, args.exp_name, time_stamp, str(cur_seed)),
            exist_ok=True,
        )
        log_training = open(
            os.path.join(
                args.log_path, args.exp_name, time_stamp, str(cur_seed), "log.csv"
            ),
            "w",
        )
        tf_writer = SummaryWriter(
            os.path.join(args.log_path, args.exp_name, time_stamp, str(cur_seed))
        )

        if args.clean_data:
            indicies = np.load("/sources/dataset/outlier_remove_idx.npy")
            train_dataset = ISLRDataSetV2(
                max_len=args.max_len,
                ver=args.preproc_ver,
                indicies=indicies,
                random_noise=args.random_noise,
                flip_x=args.flip_x,
                flip_x_v2=args.flip_x_v2,
                rotate=args.rotate,
                drop_lm=args.drop_lm,
                interpolate=args.interpolate,
            )

        else:
            train_dataset = ISLRDataSetV2(
                max_len=args.max_len,
                ver=args.preproc_ver,
                indicies=None,
                random_noise=args.random_noise,
                flip_x=args.flip_x,
                flip_x_v2=args.flip_x_v2,
                rotate=args.rotate,
                drop_lm=args.drop_lm,
                interpolate=args.interpolate,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_func,
        )

        ####### Model #######
        if args.model_ver == "v6":
            model = ISLRModelV6(
                embed_dim=args.embed_dim,
                n_head=args.n_head,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                max_len=args.max_len,
                n_layers=args.n_layers,
                input_dim=args.input_dim,
            )
        elif args.model_ver == "arcface":
            model = ISLRModelArcFace(
                embed_dim=args.embed_dim,
                n_head=args.n_head,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                cls_dropout=args.cls_dropout,
                max_len=args.max_len,
                n_layers=args.n_layers,
                input_dim=args.input_dim,
                s=args.s,
                m=args.m,
                k=args.k,
            )
        elif args.model_ver == "arcface_ce":
            model = ISLRModelArcFaceCE(
                embed_dim=args.embed_dim,
                n_head=args.n_head,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                cls_dropout=args.cls_dropout,
                max_len=args.max_len,
                n_layers=args.n_layers,
                input_dim=args.input_dim,
                s=args.s,
                m=args.m,
                k=args.k,
            )

        try:
            model = nn.DataParallel(model).cuda()
        except:
            model = model.cuda()
        cudnn.benchmark = True

        ####### Optimizer #######
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult
        )
        ####### Loss #######
        if args.loss == "ce":
            if args.weight:
                weight = np.load("/sources/dataset/weight.npy")
                weight = torch.from_numpy(weight)
            else:
                weight = None
            ce_criterion = nn.CrossEntropyLoss(
                label_smoothing=args.lb, weight=weight
            ).cuda()
            arc_criterion = nn.CrossEntropyLoss(label_smoothing=args.lb).cuda()
        else:
            raise NotImplementedError

        ####### SWA #######
        if args.swa:
            swa_model = AveragedModel(model)
            swa_start = int(args.epochs * 0.75)
            swa_scheduler = SWALR(optimizer, swa_lr=1.5e-4)

        ####### Loop #######
        for epoch in range(1, args.epochs + 1):
            if args.sd and epoch >= args.epochs // 2:
                model.module.cls_p = 0.4

            train(
                train_loader,
                model,
                ce_criterion,
                arc_criterion,
                optimizer,
                epoch,
                log_training,
                tf_writer,
            )
            if args.swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            if epoch % args.eval_freq == 0:
                save_checkpoint(
                    {
                        "state_dict": swa_model.state_dict()
                        if args.swa
                        else model.state_dict()
                    },
                    is_best=False,
                    ckpt_path=args.ckpt_path,
                    exp_name=args.exp_name,
                    time_stamp=time_stamp,
                    fold=cur_seed,
                    epoch=epoch,
                )
                tf_writer.close()


def train(
    train_loader,
    model,
    ce_criterion,
    arc_criterion,
    optimizer,
    epoch,
    log_training,
    tf_writer,
):
    losses = AverageMeter()
    accs = AverageMeter()

    model.train()
    for idx, batch in enumerate(train_loader):
        y = batch["label"].cuda()
        output = model(batch)

        if isinstance(output, tuple):
            # 1. softamx, 2. arcface
            loss = args.alpha * ce_criterion(output[0], y) + args.beta * arc_criterion(
                output[1], y
            )
            if args.alpha > args.beta:
                acc = accuracy(output[0], y)
            else:
                acc = accuracy(output[1], y)

        else:
            loss = ce_criterion(output, y)
            acc = accuracy(output, y)

        losses.update(loss.item(), y.size(0))
        accs.update(acc, y.size(0))

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
            print(info)
            log_training.write(info + "\n")
            log_training.flush()

    tf_writer.add_scalar("loss/train", losses.avg, epoch)
    tf_writer.add_scalar("acc/train", accs.avg, epoch)
    tf_writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)

    losses.reset()
    accs.reset()


def validate(
    val_loader,
    model,
    ce_criterion,
    arc_criterion,
    epoch,
    log_training,
    tf_writer,
    margin=None,
):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            y = batch["label"].cuda()
            output = model(batch)

            if isinstance(output, tuple):
                # 1. softamx, 2. arcface
                loss = args.alpha * ce_criterion(
                    output[0], y
                ) + args.beta * arc_criterion(output[1], y)

                if args.alpha > args.beta:
                    acc = accuracy(output[0], y)
                else:
                    acc = accuracy(output[1], y)

            else:
                loss = ce_criterion(output, y)
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    main()
