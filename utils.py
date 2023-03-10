import os
import torch
import shutil


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, ckpt_path, exp_name, time_stamp, fold, epoch):
    save_path = os.path.join(ckpt_path, exp_name, time_stamp, str(fold))
    file_name = f"ckpt_epoch_{epoch}.pth.tar"
    save_name = os.path.join(save_path, file_name)

    torch.save(state, save_name)
    if is_best:
        shutil.copyfile(
            save_name,
            save_name.replace(f"ckpt_epoch_{epoch}.pth.tar", "best.pth.tar"),
        )


def accuracy(output, target):
    batch_size = target.size(0)
    pred = torch.argmax(output, dim=-1)
    correct = pred.eq(target).type(torch.int64).sum() / batch_size
    return correct
