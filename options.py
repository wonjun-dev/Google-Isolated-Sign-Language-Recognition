import argparse

parser = argparse.ArgumentParser(description="Google ISLR")

parser.add_argument("--exp_name", default="test", type=str)
# ========================= Data Configs ==========================
parser.add_argument(
    "--data_root_path",
    default="/sources/dataset",
    type=str,
    help="root path of dataset",
)
parser.add_argument("--data_ver", default=0, type=int, help="version of dataset")
parser.add_argument("--folds", default=5, type=int, help="the number of folds")
# ========================= Model Configs ==========================

# ========================= Learning Configs ==========================
parser.add_argument("--epochs", default=100, type=int, help="number of total epochs")
parser.add_argument(
    "--batch_size", default=64, type=int, help="number of samples per iteration"
)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--loss", default="ce", type=str)
# ========================= Monitor Configs ==========================
parser.add_argument("--eval_freq", default=5, type=int, help="evaluation frequency")
parser.add_argument("--print_freq", default=100, type=int, help="evaluation frequency")
# ========================= Runtime Configs ==========================
parser.add_argument(
    "--workers", default=16, type=int, help="number of data loading workers"
)
parser.add_argument(
    "--log_path",
    default="/sources/log",
    type=str,
    help="log path",
)
parser.add_argument(
    "--ckpt_path",
    default="/sources/ckpts",
    type=str,
    help="ckpt path",
)
