import argparse
import gc
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.network import Network
from src.train.dual.cli import add_training_args
from src.train.dual.data import (
    make_dpa_dataset,
    make_dual_dataset,
    make_gonogo_dataset,
    split_data,
)
from src.train.dual.losses import DualLoss
from src.train.dual.trainer import (
    fit,
    freeze_for_dpa,
    freeze_for_dual,
    freeze_for_gonogo,
    print_trainable_parameters,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


TASK_SPECS = {
    "dpa": {
        "freeze": freeze_for_dpa,
        "dataset_builder": make_dpa_dataset,
        "dataset_kwargs": lambda args: {"n_batch": args.n_batch, "rank": args.rank},
        "loss_kwargs": lambda idx, args: {
            "alpha": args.bce_alpha,
            "thresh": args.loss_thresh,
            "rwd_idx": idx["rwd_idx"],
            "stim_idx": idx["stim_idx"],
            "test_idx": idx["test_idx"],
            "zero_idx": idx["zero_idx"],
            "class_bal": [1.0, 1.0],
            "read_idx": [1, 0, 2],
        },
        "zero_grad": None,
    },
    "gonogo": {
        "freeze": freeze_for_gonogo,
        "dataset_builder": make_gonogo_dataset,
        "dataset_kwargs": lambda args: {"n_batch": args.n_batch},
        "loss_kwargs": lambda idx, args: {
            "alpha": args.bce_alpha,
            "thresh": args.loss_thresh,
            "rwd_idx": idx["rwd_idx"],
            "zero_idx": idx["zero_idx"],
            "stim_idx": idx["stim_idx"],
            "class_bal": [0.0, 1.0],
            "read_idx": [1, 1],
        },
        "zero_grad": 0,
    },
    "dual": {
        "freeze": freeze_for_dual,
        "dataset_builder": make_dual_dataset,
        "dataset_kwargs": lambda args: {"n_batch": args.n_batch, "rank": args.rank},
        "loss_kwargs": lambda idx, args: {
            "alpha": args.bce_alpha,
            "thresh": args.loss_thresh,
            "stim_idx": idx["stim_idx"],
            "gng_idx": idx["gng_idx"],
            "cue_idx": idx["cue_idx"],
            "rwd_idx": idx["rwd_idx"],
            "test_idx": idx["test_idx"],
            "zero_idx": idx["zero_idx"],
            "class_bal": [1.0, 0.0, 1.0, 0.0],
            "read_idx": [1, 0, 1, 1, -1],
        },
        "zero_grad": None,
    },
}


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def resolve_device(device_arg):
    if torch.cuda.is_available() and str(device_arg).startswith("cuda"):
        return device_arg
    return "cpu"


def build_model(args):
    device = resolve_device(args.device)

    model = Network(
        args.conf_name,
        REPO_ROOT,
        VERBOSE=0,
        DEVICE=device,
        SEED=args.seed,
        N_BATCH=1,
        TRAINING=1,
    )
    model.to(torch.device(device))
    return model


def load_checkpoint(model, init_from):
    if init_from is None:
        return

    ckpt = torch.load(init_from, map_location=torch.device(model.device))
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {init_from}")


def build_task(task, model, args):
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task}")

    spec = TASK_SPECS[task]
    spec["freeze"](model)

    ff_input, labels, idx = spec["dataset_builder"](
        model=model,
        device="cpu",
        **spec["dataset_kwargs"](args),
    )

    criterion = DualLoss(
        device=model.device,
        **spec["loss_kwargs"](idx, args),
    )

    return ff_input, labels, criterion, spec["zero_grad"]


def save_checkpoint(save_path, model, args, history, task):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    metadata = {
        "task": task,
        "args": vars(args),
        "history": history,
    }
    payload = {
        "model_state_dict": model.state_dict(),
        **metadata,
    }

    torch.save(payload, save_path)
    print(f"Saved checkpoint: {save_path}")

    meta_path = save_path.replace(".pth", ".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def run_one_task(model, task, args, save_path):
    print(f"\n=== Training task: {task} ===")

    ff_input, labels, criterion, zero_grad = build_task(task, model, args)
    print("ff_input:", tuple(ff_input.shape), "labels:", tuple(labels.shape))
    print_trainable_parameters(model)

    train_loader, val_loader = split_data(
        ff_input,
        labels,
        train_perc=args.train_perc,
        batch_size=args.batch_size,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters found for task '{task}'")

    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        thresh=args.stop_loss,
        zero_grad=zero_grad,
        amp_enabled=args.amp,
    )

    save_checkpoint(save_path, model, args, history, task)

    del model, ff_input, labels
    del train_loader, val_loader, optimizer, criterion
    free_gpu()


def build_sequence_paths(outdir, seed):
    if outdir is None:
        raise ValueError("outdir must not be None")
    os.makedirs(outdir, exist_ok=True)
    return {
        "dpa": os.path.join(outdir, f"dpa_seed{seed}.pth"),
        "gonogo": os.path.join(outdir, f"gonogo_seed{seed}.pth"),
        "dual": os.path.join(outdir, f"dual_seed{seed}.pth"),
    }


def clone_args(args, **updates):
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def get_task_logfile(log_root, task, seed):
    if log_root is None:
        return None
    Path(log_root).mkdir(parents=True, exist_ok=True)
    return Path(log_root) / f"{task}_seed{seed}.log"


@contextmanager
def redirect_output_to_file(path):
    if path is None:
        yield
        return

    old_stdout, old_stderr = sys.stdout, sys.stderr
    f = open(path, "w")
    try:
        sys.stdout = f
        sys.stderr = f
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        f.close()


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", choices=["dpa", "gonogo", "dual", "sequence"], required=True)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--log-root", type=str, default=None)

    add_training_args(parser, default_n_batch=128)
    return parser


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    paths = build_sequence_paths(args.outdir, args.seed) if args.outdir is not None else None

    if args.task in TASK_SPECS:
        model = build_model(args)

        if args.save_path is None:
            raise ValueError("--save-path is required for single-task training")

        task_log = get_task_logfile(args.log_root, args.task, args.seed)
        with redirect_output_to_file(task_log):
            if paths is not None:
                if args.task == "gonogo":
                    load_checkpoint(model, paths["dpa"])

                if args.task == "dual":
                    load_checkpoint(model, paths["gonogo"])

            run_one_task(model, args.task, args, args.save_path)
        return

    if args.task == "sequence":
        if args.outdir is None:
            raise ValueError("--outdir is required for --task sequence")

        paths = build_sequence_paths(args.outdir, args.seed)
        base_n_batch = args.n_batch

        sequence = [
            ("dpa", clone_args(args, n_batch=base_n_batch), paths["dpa"]),
            ("gonogo", clone_args(args, n_batch=2 * base_n_batch), paths["gonogo"]),
            ("dual", clone_args(args, n_batch=int(2.0 * base_n_batch / 3.0)), paths["dual"]),
        ]

        for task, task_args, save_path in sequence:
            model = build_model(args)
            task_log = get_task_logfile(args.log_root, task, args.seed)

            with redirect_output_to_file(task_log):
                if task == "gonogo":
                    load_checkpoint(model, paths["dpa"])

                if task == "dual":
                    load_checkpoint(model, paths["gonogo"])

                run_one_task(model, task, task_args, save_path)

        return

    raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
