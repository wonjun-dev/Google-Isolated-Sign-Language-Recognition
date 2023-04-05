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
parser.add_argument("--folds", default=5, type=int, help="the number of folds")
# ========================= Model Configs ==========================
parser.add_argument("--model_ver", default="v2", type=str, help="version of model")
parser.add_argument("--max_len", default=80, type=int, help="max length of seqeunce")
parser.add_argument("--num_points", default=82, type=int, help="the number landmarks")
parser.add_argument(
    "--point_dim",
    default=3,
    type=int,
    help="the number of features for packseq(e.g. xyz=3)",
)
parser.add_argument(
    "--input_dim", default=912, type=int, help="the number of features for packseqv2"
)
parser.add_argument(
    "--embed_dim", default=256, type=int, help="the dim of xyz embedding"
)
parser.add_argument(
    "--n_layers", default=2, type=int, help="the number of encoder layer"
)
parser.add_argument("--n_head", default=4, type=int, help="the number of multihead")
parser.add_argument("--ff_dim", default=256, type=int, help="the dim of feedforward")
parser.add_argument("--dropout", default=0.1, type=float, help="the prob of dropout")
parser.add_argument(
    "--cls_dropout", default=0.4, type=float, help="the prob of cls_dropout"
)
# ========================= Augmentation Configs ==========================
parser.add_argument("--random_noise", default=False, type=bool)
parser.add_argument("--flip_x", default=False, type=bool)
parser.add_argument("--rotate", default=False, type=bool)
# ========================= Preproc Configs ==========================
parser.add_argument(
    "--preproc_ver", default="base", type=str, help="version of preprocessing"
)
# ========================= Learning Configs ==========================
parser.add_argument("--epochs", default=100, type=int, help="number of total epochs")
parser.add_argument(
    "--batch_size", default=64, type=int, help="number of samples per iteration"
)
parser.add_argument("--swa", default=False, type=bool)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--T_0", default=50, type=int)
parser.add_argument("--T_mult", default=1, type=int)
parser.add_argument("--weight_decay", default=1e-2, type=float)
parser.add_argument("--loss", default="ce", type=str)
parser.add_argument("--lb", default=0.0, type=float, help="label smoothing alpha")
parser.add_argument("--s", default=32, type=float, help="arcface scale")
parser.add_argument("--m", default=0.5, type=float, help="arcface margin")
parser.add_argument("--gclip", default=None, type=float)
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
