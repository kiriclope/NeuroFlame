"""MCP server for running NeuroFlame simulations."""
from __future__ import annotations

import io
import sys
import json
import base64
from hashlib import md5
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP

# ── project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path('.').resolve().parent
CONF_DIR = PROJECT_ROOT / "conf"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

from src.network import Network
from src.lr_utils import get_overlap

mcp = FastMCP("neuroflame", log_level="WARNING")


def _save_and_encode(fig, tag: str) -> dict:
    """Save figure to plots/ and return both filepath and base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)

    # Save to disk
    filename = f"{tag}.png"
    filepath = PLOT_DIR / filename
    buf.seek(0)
    filepath.write_bytes(buf.read())

    # Base64
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    return {"image_base64": img_b64, "filepath": str(filepath)}


# ── Tools ───────────────────────────────────────────────────────────

@mcp.tool()
def list_configs() -> str:
    """List all available YAML configuration files."""
    configs = sorted(p.name for p in CONF_DIR.glob("*.yml") if p.name != "defaults.yml")
    return json.dumps(configs)


@mcp.tool()
def get_config(config_name: str) -> str:
    """Return the contents of a configuration file.

    Parameters
    ----------
    config_name : str
        Filename inside the conf/ directory (e.g. 'config_test.yml').
    """
    path = CONF_DIR / config_name
    if not path.exists():
        return json.dumps({"error": f"Config '{config_name}' not found."})
    return path.read_text()


@mcp.tool()
def get_defaults() -> str:
    """Return the contents of the defaults.yml configuration."""
    path = CONF_DIR / "defaults.yml"
    return path.read_text()


@mcp.tool()
def run_simulation(
    config_name: str,
    overrides: str = "{}",
    return_overlap: bool = False,
) -> str:
    """Run a simulation and return results as JSON.

    Parameters
    ----------
    config_name : str
        Config filename (e.g. 'config_test.yml').
    overrides : str
        JSON string of parameter overrides, e.g. '{"N_BATCH": 4, "GAIN": 0.6}'.
    return_overlap : bool
        If True, also compute and return overlap with PHI0.

    Returns
    -------
    str
        JSON with keys: shape, rates_mean, rates_std, time_axis,
        and optionally overlap.
    """
    try:
        ovr = json.loads(overrides) if overrides else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid overrides JSON: {e}"})

    try:
        model = Network(config_name, str(PROJECT_ROOT), **ovr)
        cfg = model.cfg

        with torch.no_grad():
            rates = model(init_state=True)

        rates_np = rates.cpu().numpy()

        t0 = (cfg.N_STEADY + cfg.N_HEBB) * cfg.DT
        dt_window = cfg.T_WINDOW
        n_windows = rates_np.shape[1]
        time_axis = (t0 + np.arange(n_windows) * dt_window).tolist()

        result = {
            "shape": list(rates_np.shape),
            "time_axis": time_axis,
            "rates_mean": rates_np.mean(axis=(0, 2)).tolist(),
            "rates_std": rates_np.std(axis=(0, 2)).tolist(),
        }

        if return_overlap:
            overlap = get_overlap(cfg, rates)
            result["overlap"] = overlap.cpu().numpy().tolist()

        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def plot_rates(
    config_name: str,
    overrides: str = "{}",
    neuron_indices: str = "[]",
    plot_type: str = "mean",
) -> str:
    """Run a simulation and return a base64-encoded PNG plot of firing rates.

    Parameters
    ----------
    config_name : str
        Config filename.
    overrides : str
        JSON string of parameter overrides.
    neuron_indices : str
        JSON list of neuron indices to plot individually.
        Ignored when plot_type='mean'.
    plot_type : str
        'mean' — population mean ± std across neurons.
        'neurons' — individual neuron traces (uses neuron_indices).
        'heatmap' — batch-averaged heatmap of all neurons over time.

    Returns
    -------
    str
        JSON with key 'image_base64' containing the PNG.
    """
    try:
        ovr = json.loads(overrides) if overrides else {}
        idx = json.loads(neuron_indices) if neuron_indices else []
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON argument: {e}"})

    try:
        model = Network(config_name, str(PROJECT_ROOT), **ovr)
        cfg = model.cfg

        with torch.no_grad():
            rates = model(init_state=True).cpu().numpy()

        t0 = (cfg.N_STEADY + cfg.N_HEBB) * cfg.DT
        n_win = rates.shape[1]
        time = t0 + np.arange(n_win) * cfg.T_WINDOW

        fig, ax = plt.subplots(figsize=(8, 4))

        if plot_type == "heatmap":
            avg = rates.mean(axis=0)
            im = ax.imshow(
                avg.T, aspect="auto", origin="lower",
                extent=[time[0], time[-1], 0, avg.shape[1]],
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Neuron index")
            fig.colorbar(im, ax=ax, label="Rate")

        elif plot_type == "neurons":
            avg = rates.mean(axis=0)
            if not idx:
                idx = list(range(min(5, avg.shape[1])))
            for i in idx:
                ax.plot(time, avg[:, i], label=f"n{i}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Rate")
            ax.legend(fontsize=7)

        else:
            pop_mean = rates.mean(axis=(0, 2))
            pop_std = rates.std(axis=(0, 2))
            ax.plot(time, pop_mean, color="C0")
            ax.fill_between(
                time, pop_mean - pop_std, pop_mean + pop_std,
                alpha=0.25, color="C0",
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Population mean rate")

        for i, (on, off) in enumerate(zip(cfg.T_STIM_ON, cfg.T_STIM_OFF)):
            ax.axvspan(float(on), float(off), alpha=0.08, color=f"C{i + 1}")

        ax.set_title(f"{config_name} — {plot_type}")
        fig.tight_layout()

        tag = md5(f"{config_name}{overrides}{plot_type}".encode()).hexdigest()[:8]
        result = _save_and_encode(fig, f"rates_{plot_type}_{tag}")
        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def plot_overlap(
    config_name: str,
    overrides: str = "{}",
) -> str:
    """Run a simulation and return a base64-encoded PNG of the overlap with PHI0.

    Parameters
    ----------
    config_name : str
        Config filename.
    overrides : str
        JSON string of parameter overrides.
    """
    try:
        ovr = json.loads(overrides) if overrides else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    try:
        model = Network(config_name, str(PROJECT_ROOT), **ovr)
        cfg = model.cfg

        with torch.no_grad():
            rates = model(init_state=True)

        overlap = get_overlap(cfg, rates).cpu().numpy()

        t0 = (cfg.N_STEADY + cfg.N_HEBB) * cfg.DT
        n_win = overlap.shape[1]
        time = t0 + np.arange(n_win) * cfg.T_WINDOW

        fig, ax = plt.subplots(figsize=(8, 4))
        avg = overlap.mean(axis=0)

        for k in range(avg.shape[1]):
            ax.plot(time, avg[:, k], label=f"mode {k}")

        for i, (on, off) in enumerate(zip(cfg.T_STIM_ON, cfg.T_STIM_OFF)):
            ax.axvspan(float(on), float(off), alpha=0.08, color=f"C{i + 1}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Overlap")
        ax.set_title(f"Overlap — {config_name}")
        ax.legend(fontsize=7)
        fig.tight_layout()

        tag = md5(f"{config_name}{overrides}overlap".encode()).hexdigest()[:8]
        result = _save_and_encode(fig, f"overlap_{tag}")
        return json.dumps(result)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    mcp.run(transport="stdio")
