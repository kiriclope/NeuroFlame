from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import pandas as pd

from src.network import Network
from src.test.dual.task_score import DualScore
from src.test.dual.data import make_dual_test_data


def _load_checkpoint_payload(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        saved_args = payload.get("args", {})
    else:
        state_dict = payload
        saved_args = {}
    return payload, state_dict, saved_args


def _model_kwargs_from_saved_args(saved_args, default_model_kwargs=None):
    default_model_kwargs = default_model_kwargs or {}

    var_ff = saved_args.get("var_ff", None)
    if var_ff is None:
        var_ff = default_model_kwargs.get("VAR_FF", [0.25, 0.25])
    else:
        var_ff = [var_ff, var_ff]

    kwargs = {
        "TRAINING": default_model_kwargs.get("TRAINING", 1),
        "GAIN": saved_args.get("gain", default_model_kwargs.get("GAIN", 1.0)),
        "LR_INI": saved_args.get("lr_ini", default_model_kwargs.get("LR_INI", 1.0)),
        "RANK": saved_args.get("rank", default_model_kwargs.get("RANK", 2)),
        "VAR_FF": var_ff,
        "TRAIN_SCALE": saved_args.get("train_scale", default_model_kwargs.get("TRAIN_SCALE", "all")),
        "LR_TYPE": saved_args.get("lr_type", default_model_kwargs.get("LR_TYPE", "standard")),
        "K": saved_args.get("K", default_model_kwargs.get("K", 250)),
    }
    return kwargs


def load_model(
    conf_name,
    repo_root,
    device,
    seed,
    checkpoint,
    model_kwargs=None,
):
    checkpoint = Path(checkpoint)
    _, state_dict, saved_args = _load_checkpoint_payload(checkpoint, device)
    resolved_kwargs = _model_kwargs_from_saved_args(saved_args, model_kwargs)

    model = Network(
        conf_name,
        repo_root,
        VERBOSE=0,
        DEVICE=device,
        SEED=seed,
        N_BATCH=1,
        **resolved_kwargs,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.device = torch.device(device)
    model.eval()
    return model


def make_plot_readout(model, readout_cpu, rates_cpu):
    """
    Build the plotting readout used in the notebook.

    If the model already has >=3 readout channels, return readout unchanged.
    If it has exactly 2 readout channels, synthesize a 3rd 'test' channel:
        test_out = rates @ (odors[4] + odors[9]) / Na[0]
    """
    if readout_cpu.shape[-1] >= 3:
        return readout_cpu

    if (
        readout_cpu.shape[-1] == 2
        and hasattr(model, "odors")
        and hasattr(model, "Na")
    ):
        test_vec = (model.odors[4] + model.odors[9]).detach().cpu()
        test_out = (rates_cpu @ test_vec).unsqueeze(-1) / model.Na[0].detach().cpu()
        return torch.cat((readout_cpu, test_out.cpu()), dim=-1)

    return readout_cpu


def _make_time_axis(model, n_time):
    """
    Build a plotting time axis aligned with the notebook's canonical 0..10 s scale.
    """
    try:
        steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)
        steps = steps[:n_time]
        if len(steps) <= 1:
            return np.linspace(0.0, 10.0, n_time)

        return 10.0 * (steps - steps[0]) / (steps[-1] - steps[0])
    except Exception:
        return np.linspace(0.0, 10.0, n_time)


def _compute_lowrank_projections(rates_cpu, model):
    """
    Returns dict with projections onto all low-rank vectors.
    projections["U"]: [trials, time, rank]
    projections["V"]: [trials, time, rank]
    """
    out = {"U": None, "V": None}

    if not hasattr(model, "low_rank"):
        return out
    if not hasattr(model.low_rank, "U") or not hasattr(model.low_rank, "V"):
        return out

    U = model.low_rank.U.detach().cpu()
    V = model.low_rank.V.detach().cpu()

    proj_u = torch.einsum("btn,nr->btr", rates_cpu, U)
    proj_v = torch.einsum("btn,nr->btr", rates_cpu, V)

    out["U"] = proj_u
    out["V"] = proj_v
    return out


def _compute_endpoint_decisions(plot_readout_cpu, labels_cpu, idx):
    """
    Compute simple per-trial endpoint decisions from reward/cue windows.

    Returns:
        decisions: dict of per-trial predictions and targets
    """
    decisions = {}

    y = labels_cpu[..., 0]  # [trials, 4]
    rwd_idx = idx.get("rwd_idx", [])
    cue_idx = idx.get("cue_idx", [])

    # DPA / match decision from readout channel 0 over reward window
    if len(rwd_idx) > 0 and plot_readout_cpu.shape[-1] >= 1:
        dpa_signal = plot_readout_cpu[:, rwd_idx, 0].mean(dim=1)
        dpa_pred = (dpa_signal > 0).float()
        dpa_true = y[:, 0].float()

        decisions["dpa_signal"] = dpa_signal
        decisions["dpa_pred"] = dpa_pred
        decisions["dpa_true"] = dpa_true

    # DRT / go-nogo decision from readout channel 1 over cue window if present,
    # otherwise reward window as fallback.
    drt_window = cue_idx if len(cue_idx) > 0 else rwd_idx
    if len(drt_window) > 0 and plot_readout_cpu.shape[-1] >= 2:
        drt_signal = plot_readout_cpu[:, drt_window, 1].mean(dim=1)
        drt_pred = (drt_signal > 0).float()

        # test labels use ternary coding in label[2]: -1, 0, 1
        # binary go-nogo target is positive context only
        drt_true = (y[:, 2] == 1).float()

        decisions["drt_signal"] = drt_signal
        decisions["drt_pred"] = drt_pred
        decisions["drt_true"] = drt_true

    return decisions


@torch.no_grad()
def evaluate_dual_model(model, seed=None, checkpoint=None, batch_size=16):
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    ff_input, labels, idx = make_dual_test_data(
        model,
        batch_size=batch_size,
        device=device,
    )

    rates = model.forward(ff_input=ff_input).detach().cpu()
    readout = model.readout.detach().cpu()
    plot_readout = make_plot_readout(model, readout, rates)

    labels_cpu = labels.detach().cpu()

    criterion = DualScore(
        cue_idx=idx["cue_idx"],
        rwd_idx=idx["rwd_idx"],
        read_idx=[1, 1],
    )
    dpa_perf, drt_perf = criterion(readout.to(device), labels.clone())
    dpa_perf = dpa_perf.detach().cpu()
    drt_perf = drt_perf.detach().cpu()

    projections = _compute_lowrank_projections(rates, model)
    decisions = _compute_endpoint_decisions(plot_readout, labels_cpu, idx)
    time = _make_time_axis(model, plot_readout.shape[1])

    result = {
        "readout": readout,
        "plot_readout": plot_readout,
        "rates": rates,
        "labels": labels_cpu,
        "dpa_perf": dpa_perf,
        "drt_perf": drt_perf,
        "idx": idx,
        "vectors": {
            "U": model.low_rank.U.detach().cpu().numpy() if hasattr(model.low_rank, "U") else None,
            "V": model.low_rank.V.detach().cpu().numpy() if hasattr(model.low_rank, "V") else None,
            "odors": model.odors.detach().cpu().numpy() if hasattr(model, "odors") else None,
        },
        "projections": {
            "U": projections["U"],
            "V": projections["V"],
        },
        "decisions": decisions,
        "meta": {
            "seed": seed,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "Na": model.Na.detach().cpu().numpy() if hasattr(model, "Na") else None,
            "rank": getattr(model, "RANK", None),
            "time": time.tolist(),
            "n_trials": int(plot_readout.shape[0]),
            "n_time": int(plot_readout.shape[1]),
            "n_readout": int(plot_readout.shape[2]),
        },
    }
    return result


def _safe_sem(x: torch.Tensor) -> float:
    x = x.float()
    n = len(x)
    if n <= 1:
        return 0.0
    return (x.std(unbiased=True) / (n ** 0.5)).item()


def summarize_eval(result):
    labels = result["labels"]
    dpa_perf = result["dpa_perf"]
    drt_perf = result["drt_perf"]
    seed = result["meta"]["seed"]
    checkpoint = result["meta"]["checkpoint"]

    rows = []
    for task in [0, 1, -1]:
        idx = torch.where(labels[:, 2, 0] == task)[0]
        vals = dpa_perf[idx].float()
        rows.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "metric": "dpa",
                "condition": str(task),
                "mean": vals.mean().item() if len(vals) > 0 else float("nan"),
                "sem": _safe_sem(vals) if len(vals) > 0 else float("nan"),
                "n": len(vals),
            }
        )

    vals = drt_perf.float()
    rows.append(
        {
            "seed": seed,
            "checkpoint": checkpoint,
            "metric": "drt",
            "condition": "all",
            "mean": vals.mean().item() if len(vals) > 0 else float("nan"),
            "sem": _safe_sem(vals) if len(vals) > 0 else float("nan"),
            "n": len(vals),
        }
    )

    return pd.DataFrame(rows)
