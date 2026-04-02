#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.train.dual.cli import FORWARDED_TRAIN_ARGS, add_training_args, cli_name


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--task", choices=["dpa", "gonogo", "dual", "sequence"], default="sequence")
    p.add_argument("--n-runs", type=int, default=10, help="Number of seeds to launch")
    p.add_argument("--start-seed", type=int, default=1)

    p.add_argument("--python", default=sys.executable)
    p.add_argument("--module", default="src.train.dual.run_dual")
    p.add_argument("--out-root", default="runs/dual")
    p.add_argument("--log-root", default="logs/dual")

    p.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids, e.g. --gpus 0 1 2")
    p.add_argument("--max-procs-per-gpu", type=int, default=1)
    p.add_argument("--poll-seconds", type=float, default=5.0)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    add_training_args(p, default_n_batch=256)
    return p.parse_args()


def get_outdir(args, seed):
    return Path(args.out_root) / f"seed{seed}"


def get_logfile(args, seed):
    return Path(args.log_root) / f"{args.task}_seed{seed}.log"


def get_done_file(args, seed):
    outdir = get_outdir(args, seed)
    name = "dual" if args.task == "sequence" else args.task
    return outdir / f"{name}_seed{seed}.pth"


def build_single_cmd(args, seed):
    outdir = get_outdir(args, seed)
    logfile = get_logfile(args, seed)

    cmd = [
        args.python,
        "-u",
        "-m",
        args.module,
        "--task",
        args.task,
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
    ]

    for arg_name in FORWARDED_TRAIN_ARGS:
        if arg_name == "amp":
            continue
        cmd.extend([cli_name(arg_name), str(getattr(args, arg_name))])

    if args.amp:
        cmd.append("--amp")

    cmd += ["--outdir", str(outdir)]
    cmd += ["--log-root", str(args.log_root)]

    if args.task == "sequence":
        done_file = outdir / f"dual_seed{seed}.pth"
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        save_path = outdir / f"{args.task}_seed{seed}.pth"
        cmd += ["--save-path", str(save_path)]
        done_file = save_path

    return cmd, logfile, done_file


def launch_process(cmd, logfile, gpu, dry_run=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"

    logfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"[launch gpu={gpu}] {' '.join(cmd)}")
    print(f"[log] {logfile}")

    if dry_run:
        return None

    handle = None
    try:
        handle = open(logfile, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception:
        if handle is not None:
            handle.close()
        raise

    return {"proc": proc, "logfile": handle, "gpu": str(gpu), "cmd": cmd}


def cleanup_finished(active):
    remaining = []

    for item in active:
        ret = item["proc"].poll()
        if ret is None:
            remaining.append(item)
            continue

        item["logfile"].close()
        status = "OK" if ret == 0 else f"FAIL({ret})"
        print(f"[done gpu={item['gpu']}] {status} :: {' '.join(item['cmd'])}")

    return remaining


def count_gpu_load(active, gpu):
    gpu = str(gpu)
    return sum(item["gpu"] == gpu for item in active)


def main():
    args = parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_runs))
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    pending = [
        seed
        for seed in seeds
        if args.overwrite or not get_done_file(args, seed).exists()
    ]

    for seed in seeds:
        done_file = get_done_file(args, seed)
        if done_file.exists() and not args.overwrite:
            print(f"[skip] seed={seed} already done: {done_file}")

    print(f"Requested runs: {len(seeds)}")
    print(f"Pending runs:   {len(pending)}")
    print(f"GPUs:           {args.gpus}")
    print(f"Max/GPU:        {args.max_procs_per_gpu}")

    active = []
    pending_iter = iter(pending)

    while True:
        active = cleanup_finished(active)
        launched_any = False

        for gpu in args.gpus:
            while count_gpu_load(active, gpu) < args.max_procs_per_gpu:
                try:
                    seed = next(pending_iter)
                except StopIteration:
                    break

                cmd, logfile, done_file = build_single_cmd(args, seed)
                if done_file.exists() and not args.overwrite:
                    print(f"[skip] seed={seed} already done: {done_file}")
                    continue

                proc_info = launch_process(cmd, logfile, gpu, dry_run=args.dry_run)
                if proc_info is not None:
                    active.append(proc_info)
                launched_any = True

        if args.dry_run:
            break
        if not active and not launched_any:
            break

        time.sleep(args.poll_seconds)

    while active and not args.dry_run:
        active = cleanup_finished(active)
        time.sleep(args.poll_seconds)

    print("[all done]")


if __name__ == "__main__":
    main()
