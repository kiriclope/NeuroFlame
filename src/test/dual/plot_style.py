from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import seaborn as sns

# Common manuscript widths (inches)
SINGLE_COL = 3.4
ONEHALF_COL = 5.2
DOUBLE_COL = 7.0

# Consistent palettes
DUAL_PALETTE = {
    "AD": "#332288",
    "AC": "#88CCEE",
    "BD": "#117733",
    "BC": "#44AA99",
}

TASK_PALETTE = {
    "0": "#999999",
    "1": "#CC6677",
    "-1": "#4477AA",
}


def golden_height(width: float, ratio: float = (5**0.5 - 1) / 2) -> float:
    return width * ratio


def apply_plot_style(mode: str = "paper") -> None:
    sns.set_style("ticks")

    if mode == "paper":
        sns.set_context(
            "paper",
            rc={
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "lines.linewidth": 1.4,
                "lines.markersize": 4,
            },
        )
        base_font = 8
    elif mode == "talk":
        sns.set_context("talk")
        base_font = 11
    elif mode == "poster":
        sns.set_context("poster")
        base_font = 14
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mpl.rcParams["font.size"] = base_font
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["axes.titlepad"] = 8
    mpl.rcParams["axes.labelpad"] = 4

    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3
    mpl.rcParams["xtick.minor.size"] = 2
    mpl.rcParams["ytick.minor.size"] = 2

    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"

    # Better export into Illustrator / Inkscape / publisher pipelines
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    # Default, but most plots should still override explicitly
    mpl.rcParams["figure.figsize"] = (SINGLE_COL, golden_height(SINGLE_COL))


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig, outpath) -> None:
    outdir = Path(outpath).parent
    outdir.mkdir(parents=True, exist_ok=True)

    outname = Path(outpath).name
    final_path = outdir / outname
    final_path = final_path.with_suffix(".svg")

    fig.savefig(final_path, format="svg", bbox_inches="tight")


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )
