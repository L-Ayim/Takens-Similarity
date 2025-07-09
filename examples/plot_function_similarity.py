#!/usr/bin/env python3
"""
Example: synthetic similarity comparisons across very different segments.
Generates 3 segments (sin, scaled+shifted sin, noise), slides windows,
then for one ref in each segment, finds best/worst and plots + saves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from takens_similarity import (
    sliding_window_embeddings,
    compute_similarities_to_ref,
    find_best_and_worst,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Build synthetic series (600 pts total)
# ──────────────────────────────────────────────────────────────────────────────
length = 600
t = np.linspace(0, 4*np.pi, length)

seg1 = np.sin(t)                            # pure sine
seg2 = 1.5 * np.sin(t + np.pi/4)            # amplitude & phase shift
seg3 = 0.3 * np.random.randn(length)        # noise

series = np.concatenate([seg1, seg2, seg3])
x      = np.arange(series.size)            # simple x-axis

# ──────────────────────────────────────────────────────────────────────────────
# 2. Takens embedding parameters
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 50
STEP        = 50
DELAY, DIM  = 1, 2

# Precompute embeddings once
embs      = sliding_window_embeddings(series, WINDOW_SIZE, STEP, DELAY, DIM)
n_windows = len(embs)

# Select one reference window in each of the three segments:
ref_idxs = [
    0,                      # first window in sin()
    n_windows // 3,         # somewhere in seg2
    2 * n_windows // 3      # somewhere in seg3
]

# Prepare output folder
OUT_DIR = os.path.join("examples", "plots", "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Plotting routine
# ──────────────────────────────────────────────────────────────────────────────
def plot_ref(ref_idx: int):
    # 3a) compute Procrustes-based similarity scores
    sims = compute_similarities_to_ref(embs, ref_idx)
    best, best_sim, worst, worst_sim = find_best_and_worst(sims, ref_idx)

    # 3b) build full series of similarity values for coloring
    full_sims = np.zeros_like(series)
    for w, s in enumerate(sims):
        start = w * STEP
        full_sims[start:start+WINDOW_SIZE] = s

    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(0,1)

    # 3c) layout
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 3, height_ratios=[2,1], hspace=0.4, wspace=0.3)

    # Top: color‐coded time series
    ax = fig.add_subplot(gs[0, :])
    pts  = np.vstack([x, series]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, array=full_sims[:-1], cmap=cmap, norm=norm, linewidth=2)
    ax.add_collection(lc)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(series.min(), series.max())
    ax.set_title(f"Synthetic | ref={ref_idx}/{n_windows-1}, "
                 f"best={best} ({best_sim:.2f}), worst={worst} ({worst_sim:.2f})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Time index"); ax.set_ylabel("Value")
    fig.colorbar(lc, ax=ax, pad=0.02).set_label("Similarity [0–1]")

    # annotate REF in black
    r0, r1 = ref_idx*STEP, ref_idx*STEP + WINDOW_SIZE
    mid_r   = (r0 + r1)//2
    ax.plot(x[r0:r1], series[r0:r1], 'k-', lw=3)
    ax.annotate("REF",
                xy=(x[mid_r], series[mid_r]),
                xytext=(0, 30), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color="k",
                bbox=dict(fc="white", ec="k", pad=0.3),
                arrowprops=dict(arrowstyle="->", color="k", lw=2))

    # annotate BEST
    b0, b1 = best*STEP, best*STEP + WINDOW_SIZE
    mid_b   = (b0 + b1)//2
    col_b   = cmap(norm(best_sim))
    ax.plot(x[b0:b1], series[b0:b1], '--', color=col_b, lw=2)
    ax.annotate(f"BEST\n{best_sim:.2f}",
                xy=(x[mid_b], series[mid_b]),
                xytext=(0, 30), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color=col_b,
                bbox=dict(fc="white", ec=col_b, pad=0.3),
                arrowprops=dict(arrowstyle="->", color=col_b, lw=2))

    # annotate WORST
    w0, w1 = worst*STEP, worst*STEP + WINDOW_SIZE
    mid_w   = (w0 + w1)//2
    col_w   = cmap(norm(worst_sim))
    ax.plot(x[w0:w1], series[w0:w1], ':', color=col_w, lw=2)
    ax.annotate(f"WORST\n{worst_sim:.2f}",
                xy=(x[mid_w], series[mid_w]),
                xytext=(0, -30), textcoords="offset points",
                ha="center", va="top",
                fontsize=12, fontweight="bold",
                color=col_w,
                bbox=dict(fc="white", ec=col_w, pad=0.3),
                arrowprops=dict(arrowstyle="->", color=col_w, lw=2))

    # Bottom: 2D embeddings for REF / BEST / WORST
    for i,(label, idx, c) in enumerate([
        ("Window "+str(ref_idx), ref_idx, 'k'),
        ("Best "+str(best),      best,    cmap(norm(best_sim))),
        ("Worst "+str(worst),    worst,   cmap(norm(worst_sim))),
    ]):
        ax_i = fig.add_subplot(gs[1, i])
        E    = embs[idx]
        ax_i.scatter(E[:,0], E[:,1], color=c, s=30)
        ax_i.set_title(label, fontsize=12)
        ax_i.set_xlabel("Delay-0"); ax_i.set_ylabel("Delay-1")

    # save
    out = os.path.join(OUT_DIR, f"synthetic_ref{ref_idx}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved →", out)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Run for each reference
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ri in ref_idxs:
        plot_ref(ri)
