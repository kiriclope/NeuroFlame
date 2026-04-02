from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.test.dual.analysis import analyze_checkpoints


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="results/dual_analysis.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-name", type=str, default="train_dual_EI.yml")
    parser.add_argument("--repo-root", type=str, default="/home/leon/models/NeuroFlame")
    parser.add_argument("--plot-dir", type=str, default="results/dual_plots")
    return parser.parse_args()


def extract_seed(path: Path):
    m = re.search(r"seed(\d+)", path.stem)
    if m is not None:
        return int(m.group(1))

    m = re.search(r"(\d+)$", path.stem)
    if m is not None:
        return int(m.group(1))

    return None


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    ckpts = []

    for path in sorted(checkpoint_dir.glob(args.pattern)):
        seed = extract_seed(path)
        if seed is None:
            print(f"skip: could not parse seed from {path.name}")
            continue
        ckpts.append({"seed": seed, "checkpoint": path})

    if len(ckpts) == 0:
        print("No checkpoints found.")
        return

    df = analyze_checkpoints(
        checkpoints=ckpts,
        conf_name=args.conf_name,
        repo_root=args.repo_root,
        device=args.device,
        batch_size=args.batch_size,
        plot_dir=args.plot_dir,
    )

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)

    print(df)
    print(f"saved summary to {outpath}")


if __name__ == "__main__":
    main()
