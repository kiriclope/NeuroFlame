from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch

from src.test.dual.evaluate import load_model, evaluate_dual_model, summarize_eval
from src.test.dual.plot_style import apply_plot_style
from src.test.dual.plots import (
    plot_overlap_by_task,
    plot_vector_histograms,
    plot_angle_matrix,
    plot_perf_summary,
    plot_readout_heatmap,
    plot_task_condition_counts,
    plot_overlap_by_group,
)


def save_run_plots(result, plot_dir):
    seed = result["meta"]["seed"]

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(plot_dir)

    apply_plot_style(mode='paper')

    try:
        plot_overlap_by_task(result, task=0, outpath=plot_dir / f"overlap_dpa_seed{seed}.png")
        plot_overlap_by_task(result, task=1, outpath=plot_dir / f"overlap_go_seed{seed}.png")
        plot_overlap_by_task(result, task=-1, outpath=plot_dir / f"overlap_nogo_seed{seed}.png")
    except Exception as e:
        print(f"[warn] overlap plots failed for seed={seed}: {e}")

    try:
        plot_overlap_by_group(result, by="choice", task=0, outpath=plot_dir / f"overlap_choice_dpa_seed{seed}.png")
        plot_overlap_by_group(result, by="choice", task=1, outpath=plot_dir / f"overlap_choice_go_seed{seed}.png")
        plot_overlap_by_group(result, by="choice", task=-1, outpath=plot_dir / f"overlap_choice_nogo_seed{seed}.png")
        plot_overlap_by_group(result, by="choice", task=None, outpath=plot_dir / f"overlap_choice_seed{seed}.png")
    except Exception as e:
        print(f"[warn] overlap plots failed for seed={seed}: {e}")

    try:
        plot_overlap_by_group(result, by="sample", task=0, outpath=plot_dir / f"overlap_sample_dpa_seed{seed}.png")
        plot_overlap_by_group(result, by="sample", task=1, outpath=plot_dir / f"overlap_sample_go_seed{seed}.png")
        plot_overlap_by_group(result, by="sample", task=-1, outpath=plot_dir / f"overlap_sample_nogo_seed{seed}.png")
        plot_overlap_by_group(result, by="sample", task=None, outpath=plot_dir / f"overlap_sample_seed{seed}.png")
    except Exception as e:
        print(f"[warn] overlap plots failed for seed={seed}: {e}")

    try:
        plot_overlap_by_group(result, by="test", task=0, outpath=plot_dir / f"overlap_test_dpa_seed{seed}.png")
        plot_overlap_by_group(result, by="test", task=1, outpath=plot_dir / f"overlap_test_go_seed{seed}.png")
        plot_overlap_by_group(result, by="test", task=-1, outpath=plot_dir / f"overlap_test_nogo_seed{seed}.png")
        plot_overlap_by_group(result, by="test", task=None, outpath=plot_dir / f"overlap_test_seed{seed}.png")
        plot_overlap_by_group(result, by="task", task=None, outpath=plot_dir / f"overlap_task_seed{seed}.png")
    except Exception as e:
        print(f"[warn] overlap plots failed for seed={seed}: {e}")


    try:
        plot_vector_histograms(result, outpath=plot_dir / f"vectors_seed{seed}.png")
    except Exception as e:
        print(f"[warn] vector histogram failed for seed={seed}: {e}")

    try:
        plot_angle_matrix(result, outpath=plot_dir / f"angles_seed{seed}.png")
    except Exception as e:
        print(f"[warn] angle matrix failed for seed={seed}: {e}")

    try:
        plot_readout_heatmap(result, outpath=plot_dir / f"readout_heatmap_seed{seed}.png")
    except Exception as e:
        print(f"[warn] readout heatmap failed for seed={seed}: {e}")

    try:
        plot_task_condition_counts(result, outpath=plot_dir / f"condition_counts_seed{seed}.png")
    except Exception as e:
        print(f"[warn] condition count plot failed for seed={seed}: {e}")


def analyze_checkpoints(
    checkpoints,
    conf_name,
    repo_root,
    device,
    model_kwargs=None,
    batch_size=16,
    plot_dir="results/dual_plots",
):
    if plot_dir is not None:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

    dfs = []

    for item in checkpoints:
        seed = item["seed"]
        ckpt = item["checkpoint"]

        model = load_model(
            conf_name=conf_name,
            repo_root=repo_root,
            device=device,
            seed=seed,
            checkpoint=ckpt,
            model_kwargs=model_kwargs,
        )

        result = evaluate_dual_model(
            model,
            seed=seed,
            checkpoint=ckpt,
            batch_size=batch_size,
        )

        if plot_dir is not None:
            torch.save(result, Path(plot_dir) / f"eval_seed{seed}.pt")
            save_run_plots(result, plot_dir)

        df = summarize_eval(result)
        dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)

    if plot_dir is not None:
        try:
            plot_perf_summary(out, outpath=Path(plot_dir) / "perf_summary.png")
        except Exception as e:
            print(f"[warn] perf summary plot failed: {e}")

    return out
