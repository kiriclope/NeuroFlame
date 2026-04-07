import argparse


FORWARDED_TRAIN_ARGS = [
    "conf_name",
    "rank",
    "gain",
    "lr_ini",
    "var_ff",
    "train_scale",
    "lr_type",
    "K",
    "n_batch",
    "batch_size",
    "learning_rate",
    "epochs",
    "bce_alpha",
    "loss_thresh",
    "stop_loss",
    "train_perc",
    "amp",
]


def add_training_args(parser: argparse.ArgumentParser, *, default_n_batch=256):
    parser.add_argument("--conf-name", default="train_dual_EI.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")

    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--lr-ini", type=float, default=1.0)
    parser.add_argument("--var-ff", type=float, default=0.1)
    parser.add_argument("--train-scale", default="all")
    parser.add_argument("--lr-type", default="standard")
    parser.add_argument("--K", type=int, default=250)

    parser.add_argument("--n-batch", type=int, default=default_n_batch)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bce-alpha", type=float, default=1.0)
    parser.add_argument("--loss-thresh", type=float, default=5.0)
    parser.add_argument("--stop-loss", type=float, default=0.15)
    parser.add_argument("--train-perc", type=float, default=0.8)


def cli_name(arg_name: str) -> str:
    return "--" + arg_name.replace("_", "-")
