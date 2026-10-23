#!/usr/bin/env python3
"""Generate figure.

Every number plotted here is read from files in ../data:
  aigi_jpeg.o25851016      PBS job log of the submitted run (per-epoch metrics)
  dev_predictions.json     submitted development-set predictions
  test_predictions.json    submitted test-set predictions
No ground-truth dev/test labels are used (none are available).

Usage: python make_plots.py   (writes PDFs to ../figures)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

# categorical slots 1-3 of a CVD-validated palette; fixed class -> colour mapping
CLS_COL = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
CLS_NAME = {0: "Real", 1: "Model A", 2: "Model B"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})


def parse_log(path):
    pat = re.compile(r"epoch (\d+)/\d+\s+loss ([\d.]+)\s+train acc ([\d.]+)\s+"
                     r"val acc \(q\d+\) ([\d.]+)\s+val preds \[(\d+), (\d+), (\d+)\]")
    rows = [tuple(float(g) for g in m.groups()) for m in pat.finditer(Path(path).read_text())]
    return np.array(rows)


def load_preds(path):
    d = json.loads(Path(path).read_text())
    return np.array([d[f"{i}.jpg"] for i in range(len(d))])


def fig_curves(log):
    ep, loss, tr, va = log[:, 0], log[:, 1], log[:, 2], log[:, 3]
    vp = log[:, 4:7] / log[:, 4:7].sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    ax.plot(ep, tr, color=CLS_COL[0], lw=1.4, label="train acc (aug. crops)")
    ax.plot(ep, va, color=CLS_COL[1], lw=1.4, label="val acc (full, q91)")
    ax.plot(ep, loss, color=INK2, lw=1.0, ls="--", label="train loss")
    best = int(np.argmax(va))
    ax.plot(ep[best], va[best], "o", ms=3.5, color=CLS_COL[1], mec="white", mew=0.8)
    ax.annotate(f"best {va[best]:.3f}\n(ep. {int(ep[best])})", (ep[best], va[best]),
                xytext=(19, 0.55), fontsize=6, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    ax.axvspan(14.5, 15.5, color=GRID, lw=0, zorder=0)
    ax.set_xlabel("epoch"); ax.set_ylim(0, 1.15); ax.set_xlim(1, 30)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), frameon=False, handlelength=1.4,
              borderaxespad=0.1, ncol=1, labelspacing=0.15)
    ax.text(1.0, 1.08, "(a)", transform=ax.transAxes, ha="right", color=INK)
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    bottom = np.zeros_like(ep)
    for c in range(3):
        ax.bar(ep, vp[:, c], bottom=bottom, width=0.92, color=CLS_COL[c], lw=0,
               label=CLS_NAME[c])
        bottom += vp[:, c]
    ax.set_xlabel("epoch"); ax.set_ylabel("share of val preds"); ax.set_xlim(0.4, 30.6)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=1, frameon=False,
              handlelength=0.9, labelspacing=0.15)
    ax.text(1.0, 1.08, "(b)", transform=ax.transAxes, ha="right", color=INK)
    fig.savefig(FIG / "curves.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_distributions(dev, test, log):
    sets = {
        "Train (labels)": np.array([5000, 5000, 5000]),
        "Val pred. (ep. 24)": log[23, 4:7],
        "Dev pred.": np.bincount(dev, minlength=3),
        "Test pred.": np.bincount(test, minlength=3),
    }
    fig, ax = plt.subplots(figsize=(3.35, 1.25))
    names = list(sets)
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        counts = np.asarray(sets[n], float)
        share = counts / counts.sum()
        left = 0.0
        for c in range(3):
            ax.barh(yi, share[c], left=left, height=0.62, color=CLS_COL[c], lw=0,
                    label=CLS_NAME[c] if yi == y[0] else None)
            if c == 2 and 0 < share[c] <= 0.07:
                ax.annotate(f"{int(counts[c])}", (left + share[c] / 2, yi + 0.31),
                            xytext=(0.9, yi + 0.5), fontsize=6, color=INK, ha="right",
                            va="center", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
            if share[c] > 0.07:
                ax.text(left + share[c] / 2, yi, f"{int(counts[c])}", ha="center",
                        va="center", fontsize=6, color="white")
            left += share[c] + 0.004
        ax.text(1.03, yi, f"n={int(counts.sum())}", va="center", fontsize=6, color=INK2)
    ax.set_yticks(y, names); ax.set_xlim(0, 1.0); ax.set_xlabel("share of images")
    ax.axvline(1 / 3, color=INK2, lw=0.5, ls=":"); ax.axvline(2 / 3, color=INK2, lw=0.5, ls=":")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.3), ncol=3, frameon=False,
              handlelength=0.9)
    fig.savefig(FIG / "distributions.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    log = parse_log(DATA / "aigi_jpeg.o25851016")
    assert log.shape == (30, 7), log.shape
    dev = load_preds(DATA / "dev_predictions.json")
    test = load_preds(DATA / "test_predictions.json")
    fig_curves(log)
    fig_distributions(dev, test, log)
    summary = {
        "epochs_logged": int(log.shape[0]),
        "best_val_acc": float(log[:, 3].max()),
        "best_epoch": int(log[np.argmax(log[:, 3]), 0]),
        "final_train_acc": float(log[-1, 2]),
        "val_acc_ep14_ep15_ep16": log[13:16, 3].tolist(),
        "dev_pred_counts": np.bincount(dev, minlength=3).tolist(),
        "test_pred_counts": np.bincount(test, minlength=3).tolist(),
        "test_pred_counts_per_150_block": [np.bincount(test[i:i + 150], minlength=3).tolist()
                                           for i in range(0, len(test), 150)],
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
