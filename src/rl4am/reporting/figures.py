from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl4am-mpl")

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import MaxNLocator
import pandas as pd

from rl4am.reporting.comparison import align_equity_series, rebase_equity_series
from rl4am.results import load_strategy_result


def build_equity_comparison_figure(
    left_path: str | Path,
    right_path: str | Path,
    output_dir: str | Path,
    use_net: bool = True,
) -> Path:
    """Build and save an aligned equity comparison figure."""
    _configure_style()
    left = load_strategy_result(left_path)
    right = load_strategy_result(right_path)
    aligned = align_equity_series(left, right, use_net=use_net)
    aligned = rebase_equity_series(aligned)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "equity_compare.png"

    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    ax.plot(aligned.index, aligned["left"], label=left.name, color="tab:red", lw=1.0)
    ax.plot(
        aligned.index,
        aligned["right"],
        label=right.name,
        color="tab:blue",
        lw=1.0,
    )
    ax.set_title("Equity Comparison" + (" (Net)" if use_net else " (Gross)"))
    ax.set_xlabel("Date" if aligned.index.name == "date" else "Step")
    ax.set_ylabel("Equity")
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=7,
        loc="upper left",
        handlelength=2.0,
    )
    _format_x_axis(ax, aligned.index)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _configure_style() -> None:
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7


def _format_x_axis(ax, index: pd.Index) -> None:
    if isinstance(index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
