from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, t

from src.test.dual.plot_style import (
    SINGLE_COL,
    ONEHALF_COL,
    DOUBLE_COL,
    DUAL_PALETTE,
    TASK_PALETTE,
    golden_height,
    save_figure,
)


# ---------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------

def add_vlines(ax=None):
    t_STIM = [1, 2]
    t_DIST = [3, 4]
    t_CUE = [5, 5.5]
    t_TEST = [7, 8]

    periods = [t_STIM, t_DIST, t_TEST, t_CUE]
    colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#55A868"]

    if ax is None:
        ax = plt.gca()

    for period, color in zip(periods, colors):
        ax.axvspan(period[0], period[1], alpha=0.08, color=color, lw=0)


def mean_ci(data):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[None, :]

    mean = np.nanmean(data, axis=0)
    if data.shape[0] <= 1:
        return mean, np.zeros_like(mean)

    serr = sem(data, axis=0, nan_policy="omit")
    n = np.sum(~np.isnan(data), axis=0)
    df = np.maximum(n - 1, 1)
    t_val = t.ppf(0.975, df=df)
    ci = t_val * serr
    ci = np.nan_to_num(ci, nan=0.0, posinf=0.0, neginf=0.0)
    return mean, ci


def angle_AB(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    A_norm = A / (np.linalg.norm(A) + 1e-8)
    B_norm = B / (np.linalg.norm(B) + 1e-8)
    dot = np.clip(A_norm @ B_norm, -1.0, 1.0)
    return np.arccos(dot) * 180.0 / np.pi


def cosine_AB(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.clip(
        (A @ B) / ((np.linalg.norm(A) + 1e-8) * (np.linalg.norm(B) + 1e-8)),
        -1.0,
        1.0,
    )


def _panel_names(n_panels):
    names = ["Sample overlap", "Choice overlap", "Test overlap"]
    if n_panels <= len(names):
        return names[:n_panels]
    return names + [f"Readout {k}" for k in range(len(names), n_panels)]


def _get_plot_readout(result):
    readout = result["plot_readout"]
    if hasattr(readout, "detach"):
        readout = readout.detach().cpu().numpy()
    return np.asarray(readout)


def _get_rates(result):
    rates = result["rates"]
    if hasattr(rates, "detach"):
        rates = rates.detach().cpu().numpy()
    return np.asarray(rates)


def _get_labels_matrix(result):
    labels = result["labels"]
    if hasattr(labels, "detach"):
        return labels[..., 0].detach().cpu().numpy().T.copy()
    return np.asarray(labels)[..., 0].T.copy()


def _get_time_axis(result, n_time):
    meta = result.get("meta", {})
    time = meta.get("time", None)
    if time is not None:
        time = np.asarray(time)
        if len(time) == n_time:
            return time
    return np.linspace(0, 10, n_time)


def _condition_names():
    return ["AD", "AC", "BD", "BC"]


def _condition_masks(y, task):
    """
    Returns dict:
        condition_name -> boolean mask over trials
    Using the same logic as the notebook overlap plots.
    """
    masks = {}
    c = 0
    for j in range(2):      # sample
        for i in range(2):  # test
            mask = (
                (y[0] == (i == j)) &
                (y[1] == i) &
                (y[2] == task) &
                (y[3] == j)
            )
            masks[_condition_names()[c]] = mask
            c += 1
    return masks


def _project_rates(rates, vec):
    """
    rates: [trials, time, neurons]
    vec:   [neurons]
    returns [trials, time]
    """
    return rates @ vec


# ---------------------------------------------------------------------
# main existing plots
# ---------------------------------------------------------------------
#+begin_src python
def _group_masks(y, by, task=None):
    """
    Returns an ordered dict-like mapping:
        group_name -> boolean mask over trials

    Label convention:
        y[0] = choice (0/1; unpair/pair)
        y[1] = test   (0/1; second odor)
        y[2] = task   (0/1/-1)
        y[3] = sample (0/1; first odor)
    """
    by = by.lower()

    if by == "condition":
        if task is None:
            raise ValueError("task must be provided when by='condition'")
        return _condition_masks(y, task)

    masks = {}

    base = np.ones(y.shape[1], dtype=bool)
    if task is not None:
        base = base & (y[2] == task)

    if by in ("sample", "first", "sample_odor", "first_odor"):
        for v, name in zip([0, 1], ["A", "B"]):
            masks[name] = base & (y[1] == v)

    elif by in ("choice", "pairing", "paired"):
        for v, name in zip([0, 1], ["unpair", "pair"]):
            masks[name] = base & (y[0] == v)

    elif by in ("test", "second", "test_odor", "second_odor"):
        for v, name in zip([0, 1], ["C", "D"]):
            masks[name] = base & (y[3] == v)

    elif by == "task":
        for v in [0, 1, -1]:
            masks[str(v)] = (y[2] == v)

    else:
        raise ValueError(f"Unknown grouping: {by}")

    return masks


def _group_palette(by, labels):
    by = by.lower()

    if by in ("sample", "first", "sample_odor", "first_odor"):
        palette = {"A": "#4C72B0", "B": "#DD8452"}
    elif by in ("choice", "pairing", "paired"):
        palette = {"unpair": "#C44E52", "pair": "#55A868"}
    elif by in ("test", "second", "test_odor", "second_odor"):
        palette = {"C": "#8172B2", "D": "#937860"}
    elif by == "task":
        palette = {k: TASK_PALETTE[k] for k in ["0", "1", "-1"] if k in TASK_PALETTE}
    elif by == "condition":
        palette = {k: DUAL_PALETTE[k] for k in labels}
    else:
        palette = {lab: None for lab in labels}

    return [palette.get(lab, None) for lab in labels]


def plot_overlap_by_group(result, by="sample", task=None, outpath=None):
    """
    Plot overlap traces averaged by a chosen grouping.

    Parameters
    ----------
    by : str
        One of:
            'sample'   : average by first odor
            'choice'   : average by pair vs unpair
            'test'     : average by second odor
            'task'     : average by task
            'condition': original AD/AC/BD/BC grouping, requires task
    task : int or None
        If provided for sample/choice/test, restrict trials to that task.
        For by='condition', this is required.
    """
    readout = _get_plot_readout(result)
    y = _get_labels_matrix(result)

    n_panels = readout.shape[2]
    panel_names = _panel_names(n_panels)
    time = _get_time_axis(result, readout.shape[1])

    groups = _group_masks(y, by=by, task=task)
    labels = list(groups.keys())
    colors = _group_palette(by, labels)

    fig, ax = plt.subplots(
        1,
        n_panels,
        figsize=(DOUBLE_COL, 2.3),
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        ax = [ax]

    global_absmax = np.nanmax(np.abs(readout))
    if not np.isfinite(global_absmax) or global_absmax == 0:
        global_absmax = 1.0

    for k in range(n_panels):
        for label, color in zip(labels, colors):
            mask = groups[label]
            data = readout[mask, :, k]
            if data.shape[0] == 0:
                continue

            mean, ci = mean_ci(data)
            ax[k].plot(time, mean, color=color, label=label, lw=1.5)
            ax[k].fill_between(
                time,
                mean - ci,
                mean + ci,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

        add_vlines(ax[k])
        ax[k].axhline(0, color="k", ls="--", lw=0.8)
        ax[k].set_xlabel("Time (s)")
        ax[k].set_title(panel_names[k])
        ax[k].set_ylim(-1.05 * global_absmax, 1.05 * global_absmax)

    ax[0].set_ylabel("Overlap")

    title = by if task is None else f"{by}, task={task}"
    ax[0].legend(frameon=False, fontsize=7, loc="best", title=title)

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig



def plot_overlap_by_task(result, task=0, outpath=None):
    readout = _get_plot_readout(result)
    y = _get_labels_matrix(result)

    n_panels = readout.shape[2]
    panel_names = _panel_names(n_panels)

    fig, ax = plt.subplots(
        1,
        n_panels,
        figsize=(DOUBLE_COL, 2.3),
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        ax = [ax]

    time = _get_time_axis(result, readout.shape[1])

    cond_labels = _condition_names()
    colors = [
        DUAL_PALETTE["AD"],
        DUAL_PALETTE["AC"],
        DUAL_PALETTE["BD"],
        DUAL_PALETTE["BC"],
    ]

    global_absmax = np.nanmax(np.abs(readout))
    if not np.isfinite(global_absmax) or global_absmax == 0:
        global_absmax = 1.0

    for k in range(n_panels):
        for idx, (cond_name, mask) in enumerate(_condition_masks(y, task).items()):
            data = readout[mask, :, k]
            if data.shape[0] == 0:
                continue

            mean, ci = mean_ci(data)
            color = colors[idx]

            ax[k].plot(time, mean, color=color, label=cond_name, lw=1.5)
            ax[k].fill_between(
                time,
                mean - ci,
                mean + ci,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

        add_vlines(ax[k])
        ax[k].axhline(0, color="k", ls="--", lw=0.8)
        ax[k].set_xlabel("Time (s)")
        ax[k].set_title(panel_names[k])
        ax[k].set_ylim(-1.05 * global_absmax, 1.05 * global_absmax)

    ax[0].set_ylabel("Overlap")
    handles, labels_legend = ax[0].get_legend_handles_labels()
    if len(handles) > 0:
        ax[0].legend(frameon=False, fontsize=7, loc="best")

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_vector_histograms(result, outpath=None):
    U = result["vectors"]["U"]
    V = result["vectors"]["V"]

    if U is None or V is None:
        return None

    rank = U.shape[1]
    fig, ax = plt.subplots(
        1,
        rank,
        figsize=(max(ONEHALF_COL, 2.2 * rank), 2.2),
        constrained_layout=True,
    )
    if rank == 1:
        ax = [ax]

    for i in range(rank):
        ax[i].hist(U[:, i], bins="auto", histtype="step", density=True, label="m")
        ax[i].hist(V[:, i], bins="auto", histtype="step", density=True, label="n")
        ax[i].axvline(0, color="k", ls="--", lw=0.8)
        ax[i].set_title(f"dim {i}")
        ax[i].set_ylabel("Density")
        ax[i].legend(frameon=False)

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_angle_matrix(result, outpath=None):
    U = result["vectors"]["U"]
    V = result["vectors"]["V"]

    if U is None or V is None:
        return None

    rank = U.shape[1]
    if rank < 2:
        return None

    labels = []
    vectors = []
    for i in range(rank):
        vectors.extend([U[:, i], V[:, i]])
        labels.extend([f"m{i}", f"n{i}"])

    n = len(vectors)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = angle_AB(vectors[i], vectors[j])

    mask = np.triu(np.ones_like(mat, dtype=bool))
    masked = np.ma.masked_array(mat, mask=mask)

    fig, ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL),
        constrained_layout=True,
    )
    im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=180)
    ax.set_xticks(np.arange(n), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n), labels=labels)
    ax.invert_yaxis()

    for i in range(n):
        for j in range(i + 1):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Angle (deg)")

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_perf_summary(df, outpath=None):
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(ONEHALF_COL, 2.2),
        constrained_layout=True,
    )

    dpa = df[df["metric"] == "dpa"]
    drt = df[df["metric"] == "drt"]

    task_order = ["0", "1", "-1"]
    for i, task in enumerate(task_order):
        sub = dpa[dpa["condition"] == task]
        if len(sub) == 0:
            continue

        x = np.full(len(sub), i, dtype=float)
        ax[0].scatter(
            x,
            sub["mean"].values,
            s=16,
            alpha=0.6,
            color=TASK_PALETTE[task],
            zorder=2,
        )

        y = sub["mean"].mean()
        yerr = sub["mean"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
        ax[0].errorbar(
            i, y, yerr=yerr, fmt="o", color="k", capsize=2, zorder=3
        )

    ax[0].set_xticks(range(len(task_order)), task_order)
    ax[0].set_xlabel("Task")
    ax[0].set_ylabel("DPA accuracy")
    ax[0].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[0].set_ylim(0.45, 1.02)

    if len(drt) > 0:
        ax[1].scatter(
            np.zeros(len(drt)),
            drt["mean"].values,
            s=16,
            alpha=0.6,
            color="0.4",
            zorder=2,
        )
        y = drt["mean"].mean()
        yerr = drt["mean"].std(ddof=1) / np.sqrt(len(drt)) if len(drt) > 1 else 0.0
        ax[1].errorbar(0, y, yerr=yerr, fmt="o", color="k", capsize=2, zorder=3)

    ax[1].set_xticks([0], ["all"])
    ax[1].set_ylabel("GNG accuracy")
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[1].set_ylim(0.45, 1.02)

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_readout_heatmap(result, outpath=None):
    readout = _get_plot_readout(result)
    mean_readout = readout.mean(axis=0).T

    vmax = np.nanmax(np.abs(mean_readout))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(
        figsize=(ONEHALF_COL, 1.6 + 0.35 * mean_readout.shape[0]),
        constrained_layout=True,
    )
    im = ax.imshow(
        mean_readout,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Readout dim")
    ax.set_yticks(
        np.arange(mean_readout.shape[0]),
        labels=[f"r{i}" for i in range(mean_readout.shape[0])],
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean readout")

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_task_condition_counts(result, outpath=None):
    y = _get_labels_matrix(result)

    tasks = [0, 1, -1]
    counts = np.zeros((len(tasks), 4), dtype=int)

    for ti, task in enumerate(tasks):
        for ci, (_, mask) in enumerate(_condition_masks(y, task).items()):
            counts[ti, ci] = int(mask.sum())

    fig, ax = plt.subplots(
        figsize=(SINGLE_COL, golden_height(SINGLE_COL) + 0.3),
        constrained_layout=True,
    )
    im = ax.imshow(counts, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(4), labels=_condition_names())
    ax.set_yticks(np.arange(3), labels=["0", "1", "-1"])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Task")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=7)

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------
# new figures
# ---------------------------------------------------------------------

def plot_accuracy_by_condition(result, outpath=None):
    """
    Per-condition DPA endpoint accuracy estimated from final reward readout sign.
    """
    readout = _get_plot_readout(result)
    y = _get_labels_matrix(result)
    idx = result["idx"]
    rwd_idx = idx["rwd_idx"]

    if len(rwd_idx) == 0:
        return None

    # channel 0 is treated as DPA/match readout in your setup
    final_signal = readout[:, rwd_idx, 0].mean(axis=1)
    pred = (final_signal > 0).astype(float)
    true = y[0].astype(float)

    tasks = [0, 1, -1]
    fig, ax = plt.subplots(
        1, len(tasks),
        figsize=(DOUBLE_COL, 2.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(tasks) == 1:
        ax = [ax]

    for ti, task in enumerate(tasks):
        masks = _condition_masks(y, task)
        means = []
        for cond in _condition_names():
            mask = masks[cond]
            if mask.sum() == 0:
                means.append(np.nan)
            else:
                means.append(np.mean(pred[mask] == true[mask]))

        ax[ti].bar(
            np.arange(4),
            means,
            color=[DUAL_PALETTE[c] for c in _condition_names()],
            alpha=0.85,
        )
        ax[ti].axhline(0.5, color="k", ls="--", lw=0.8)
        ax[ti].set_xticks(np.arange(4), labels=_condition_names(), rotation=45)
        ax[ti].set_ylim(0.0, 1.05)
        ax[ti].set_title(f"task={task}")
        ax[ti].set_ylabel("Accuracy")

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_readout_endpoint_distributions(result, task=0, outpath=None):
    """
    Distribution of endpoint readout values by condition.
    """
    readout = _get_plot_readout(result)
    y = _get_labels_matrix(result)
    idx = result["idx"]
    rwd_idx = idx["rwd_idx"]

    if len(rwd_idx) == 0:
        return None

    n_panels = readout.shape[2]
    fig, ax = plt.subplots(
        1, n_panels,
        figsize=(DOUBLE_COL, 2.2),
        sharey=False,
        constrained_layout=True,
    )
    if n_panels == 1:
        ax = [ax]

    masks = _condition_masks(y, task)

    for k in range(n_panels):
        xs = []
        cols = []
        pos = []
        for i, cond in enumerate(_condition_names()):
            mask = masks[cond]
            vals = readout[mask][:, rwd_idx, k].mean(axis=1) if mask.sum() > 0 else np.array([])
            if len(vals) > 0:
                xs.append(vals)
                cols.append(DUAL_PALETTE[cond])
                pos.append(i)

        if len(xs) == 0:
            continue

        bp = ax[k].boxplot(xs, positions=pos, patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(bp["boxes"], cols):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)

        ax[k].axhline(0, color="k", ls="--", lw=0.8)
        ax[k].set_xticks(np.arange(4), labels=_condition_names(), rotation=45)
        ax[k].set_title(_panel_names(n_panels)[k])
        ax[k].set_ylabel("Endpoint readout")

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_lowrank_projections(result, task=0, outpath=None):
    """
    Project rates onto U and V vectors over time.
    """
    rates = _get_rates(result)
    y = _get_labels_matrix(result)
    U = result["vectors"]["U"]
    V = result["vectors"]["V"]

    if U is None or V is None:
        return None

    rank = U.shape[1]
    time = _get_time_axis(result, rates.shape[1])

    fig, ax = plt.subplots(
        2, rank,
        figsize=(max(ONEHALF_COL, 2.2 * rank), 4.0),
        sharex=True,
        constrained_layout=True,
    )
    if rank == 1:
        ax = np.array(ax).reshape(2, 1)

    masks = _condition_masks(y, task)
    conds = _condition_names()

    for i in range(rank):
        proj_u = _project_rates(rates, U[:, i])
        proj_v = _project_rates(rates, V[:, i])

        for cond in conds:
            mask = masks[cond]
            if mask.sum() == 0:
                continue

            mu, cu = mean_ci(proj_u[mask])
            mv, cv = mean_ci(proj_v[mask])

            ax[0, i].plot(time, mu, color=DUAL_PALETTE[cond], label=cond)
            ax[0, i].fill_between(time, mu - cu, mu + cu, color=DUAL_PALETTE[cond], alpha=0.15, linewidth=0)

            ax[1, i].plot(time, mv, color=DUAL_PALETTE[cond], label=cond)
            ax[1, i].fill_between(time, mv - cv, mv + cv, color=DUAL_PALETTE[cond], alpha=0.15, linewidth=0)

        add_vlines(ax[0, i])
        add_vlines(ax[1, i])
        ax[0, i].axhline(0, color="k", ls="--", lw=0.8)
        ax[1, i].axhline(0, color="k", ls="--", lw=0.8)
        ax[0, i].set_title(f"m{i}")
        ax[1, i].set_title(f"n{i}")
        ax[1, i].set_xlabel("Time (s)")

    ax[0, 0].set_ylabel("Projection")
    ax[1, 0].set_ylabel("Projection")

    handles, labels_legend = ax[0, 0].get_legend_handles_labels()
    if len(handles) > 0:
        ax[0, 0].legend(frameon=False, fontsize=7)

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_latent_trajectories(result, task=0, dims=(0, 1), basis="U", outpath=None):
    """
    2D trajectories in low-rank coordinates.
    basis: 'U' or 'V'
    """
    rates = _get_rates(result)
    y = _get_labels_matrix(result)
    vecs = result["vectors"]["U"] if basis.upper() == "U" else result["vectors"]["V"]

    if vecs is None:
        return None

    d0, d1 = dims
    if vecs.shape[1] <= max(d0, d1):
        return None

    p0 = _project_rates(rates, vecs[:, d0])
    p1 = _project_rates(rates, vecs[:, d1])

    fig, ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL),
        constrained_layout=True,
    )

    masks = _condition_masks(y, task)
    for cond in _condition_names():
        mask = masks[cond]
        if mask.sum() == 0:
            continue

        x = p0[mask].mean(axis=0)
        z = p1[mask].mean(axis=0)

        ax.plot(x, z, color=DUAL_PALETTE[cond], lw=1.5, label=cond)
        ax.scatter(x[0], z[0], color=DUAL_PALETTE[cond], s=12, zorder=3)
        ax.scatter(x[-1], z[-1], color=DUAL_PALETTE[cond], s=18, marker="s", zorder=3)

    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(f"{basis.lower()}{d0} projection")
    ax.set_ylabel(f"{basis.lower()}{d1} projection")
    ax.legend(frameon=False, fontsize=7)

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


def plot_readout_raster(result, sort_task_order=(0, 1, -1), outpath=None):
    """
    Trial-by-trial raster of plot_readout, sorted by task and condition.
    """
    readout = _get_plot_readout(result)
    y = _get_labels_matrix(result)
    n_panels = readout.shape[2]

    sort_key = []
    for task in sort_task_order:
        masks = _condition_masks(y, task)
        for cond in _condition_names():
            idx = np.where(masks[cond])[0]
            sort_key.extend(idx.tolist())

    if len(sort_key) == 0:
        return None

    sorted_readout = readout[sort_key]

    fig, ax = plt.subplots(
        n_panels, 1,
        figsize=(ONEHALF_COL, max(2.2, 1.2 * n_panels)),
        sharex=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        ax = [ax]

    for k in range(n_panels):
        dat = sorted_readout[:, :, k]
        vmax = np.nanmax(np.abs(dat))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        im = ax[k].imshow(dat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[k].set_ylabel(f"r{k}")
        fig.colorbar(im, ax=ax[k], shrink=0.7)

    ax[-1].set_xlabel("Time bin")

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------
# metrics extraction / geometry-behavior summary
# ---------------------------------------------------------------------

def extract_geometry_metrics(result):
    """
    Return one-row dict of geometry summary statistics for a single run.
    """
    meta = result.get("meta", {})
    U = result["vectors"]["U"]
    V = result["vectors"]["V"]

    row = {
        "seed": meta.get("seed", None),
        "checkpoint": meta.get("checkpoint", None),
    }

    if U is None or V is None:
        return row

    rank = U.shape[1]
    row["rank"] = rank

    for i in range(rank):
        row[f"U_norm_{i}"] = float(np.linalg.norm(U[:, i]))
        row[f"V_norm_{i}"] = float(np.linalg.norm(V[:, i]))
        row[f"angle_UV_{i}"] = float(angle_AB(U[:, i], V[:, i]))
        row[f"cosine_UV_{i}"] = float(cosine_AB(U[:, i], V[:, i]))

    for i in range(rank):
        for j in range(i + 1, rank):
            row[f"angle_U_{i}_{j}"] = float(angle_AB(U[:, i], U[:, j]))
            row[f"angle_V_{i}_{j}"] = float(angle_AB(V[:, i], V[:, j]))

    return row


def plot_geometry_behavior_scatter(df, x, y="mean", hue=None, outpath=None):
    """
    Generic scatter plot for geometry-behavior relationships.

    Example:
        plot_geometry_behavior_scatter(df, x="angle_UV_0", y="dual_dpa_mean")
    """
    fig, ax = plt.subplots(
        figsize=(SINGLE_COL, golden_height(SINGLE_COL)),
        constrained_layout=True,
    )

    if hue is not None and hue in df.columns:
        groups = df.groupby(hue)
        for name, sub in groups:
            ax.scatter(sub[x], sub[y], s=18, alpha=0.75, label=str(name))
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.scatter(df[x], df[y], s=18, alpha=0.75, color="k")

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------
# utilities for turning many single-run results into dataframes
# ---------------------------------------------------------------------

def make_condition_accuracy_table(results):
    """
    Build a dataframe with one row per seed x task x condition.
    """
    rows = []

    for result in results:
        readout = _get_plot_readout(result)
        y = _get_labels_matrix(result)
        idx = result["idx"]
        seed = result["meta"].get("seed", None)
        checkpoint = result["meta"].get("checkpoint", None)

        rwd_idx = idx["rwd_idx"]
        if len(rwd_idx) == 0:
            continue

        final_signal = readout[:, rwd_idx, 0].mean(axis=1)
        pred = (final_signal > 0).astype(float)
        true = y[0].astype(float)

        for task in [0, 1, -1]:
            masks = _condition_masks(y, task)
            for cond in _condition_names():
                mask = masks[cond]
                acc = np.mean(pred[mask] == true[mask]) if mask.sum() > 0 else np.nan
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "task": str(task),
                        "condition": cond,
                        "accuracy": acc,
                        "n": int(mask.sum()),
                    }
                )

    return pd.DataFrame(rows)


def plot_condition_accuracy_summary(df, outpath=None):
    """
    Aggregate accuracy-by-condition table across seeds.
    """
    tasks = ["0", "1", "-1"]
    conds = _condition_names()

    fig, ax = plt.subplots(
        1, len(tasks),
        figsize=(DOUBLE_COL, 2.3),
        sharey=True,
        constrained_layout=True,
    )
    if len(tasks) == 1:
        ax = [ax]

    for ti, task in enumerate(tasks):
        sub = df[df["task"] == task]
        for ci, cond in enumerate(conds):
            vals = sub[sub["condition"] == cond]["accuracy"].dropna().values
            if len(vals) == 0:
                continue

            x = np.full(len(vals), ci, dtype=float)
            ax[ti].scatter(x, vals, s=14, alpha=0.6, color=DUAL_PALETTE[cond])
            mean = vals.mean()
            err = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ax[ti].errorbar(ci, mean, yerr=err, fmt="o", color="k", capsize=2)

        ax[ti].axhline(0.5, color="k", ls="--", lw=0.8)
        ax[ti].set_xticks(np.arange(len(conds)), labels=conds, rotation=45)
        ax[ti].set_ylim(0.0, 1.05)
        ax[ti].set_title(f"task={task}")
        ax[ti].set_ylabel("Accuracy")

    if outpath is not None:
        save_figure(fig, outpath)
        plt.close(fig)
    return fig
