#!/usr/bin/env python3
"""
Example: cycle through every reference window (size=100, step=100)
using AAPL data from 2020-01-01 up to today, with Procrustes-based similarity.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from datetime import date

from takens_similarity import (
    sliding_window_embeddings,
    compute_similarities_to_ref,
    find_best_and_worst,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Configuration
# ──────────────────────────────────────────────────────────────────────────────
TICKER       = "AAPL"
START_DATE   = "2020-01-01"
END_DATE     = date.today().isoformat()
INTERVAL     = "1d"
WINDOW_SIZE  = 100
STEP         = 100
DELAY, DIM   = 1, 2

OUT_DIR = os.path.join("examples", "plots", TICKER)
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Fetch Data
# ──────────────────────────────────────────────────────────────────────────────
def fetch_prices(ticker, start, end, interval="1d"):
    df = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        actions=False
    )
    if df.empty:
        raise RuntimeError(f"No data for {ticker} from {start} to {end}")
    dates  = mdates.date2num(df.index.to_pydatetime())
    prices = df["Close"].astype(float).values
    return dates, prices

dates, prices = fetch_prices(TICKER, START_DATE, END_DATE, INTERVAL)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Precompute Takens embeddings
# ──────────────────────────────────────────────────────────────────────────────
embs = sliding_window_embeddings(prices, WINDOW_SIZE, STEP, DELAY, DIM)
n_windows = len(embs)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Plotting routine (uses Procrustes similarity)
# ──────────────────────────────────────────────────────────────────────────────
def plot_ref_window(ref_idx: int):
    # compute 0–1 similarity based on Procrustes
    sims = compute_similarities_to_ref(embs, ref_idx)
    best_idx, best_sim, worst_idx, worst_sim = find_best_and_worst(sims, ref_idx)

    # build full-length similarity array
    full_sims = np.zeros_like(prices)
    for w, sim in enumerate(sims):
        full_sims[w*STEP : w*STEP + WINDOW_SIZE] = sim

    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(0, 1)

    fig = plt.figure(figsize=(16,10))
    gs  = fig.add_gridspec(2,3, height_ratios=[2,1], hspace=0.4, wspace=0.3)

    # Top: colored time series
    ax = fig.add_subplot(gs[0,:])
    pts  = np.vstack([dates, prices]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, array=full_sims[:-1], cmap=cmap, norm=norm, linewidth=2)
    ax.add_collection(lc)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(prices.min(), prices.max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    ax.set_title(
        f"{TICKER} | ref={ref_idx}/{n_windows-1}, "
        f"best={best_idx} ({best_sim:.2f}), worst={worst_idx} ({worst_sim:.2f})",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Date"); ax.set_ylabel("Close Price")
    fig.colorbar(lc, ax=ax, pad=0.02).set_label("Procrustes similarity [0–1]")

    # Annotate REF in black
    r0, r1 = ref_idx*STEP, ref_idx*STEP + WINDOW_SIZE
    mid_r = (r0 + r1)//2
    ax.plot(dates[r0:r1], prices[r0:r1], 'k-', lw=3)
    ax.annotate(
        "REF",
        xy=(dates[mid_r], prices[mid_r]),
        xytext=(0, 30), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        color="k",
        bbox=dict(fc="white", ec="k", pad=0.3),
        arrowprops=dict(arrowstyle="->", color="k", lw=2)
    )

    # Annotate BEST
    b0, b1 = best_idx*STEP, best_idx*STEP + WINDOW_SIZE
    mid_b = (b0 + b1)//2
    col_b = cmap(norm(best_sim))
    ax.plot(dates[b0:b1], prices[b0:b1], '--', color=col_b, lw=2)
    ax.annotate(
        f"BEST\n{best_sim:.3f}",
        xy=(dates[mid_b], prices[mid_b]),
        xytext=(0, 30), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        color=col_b,
        bbox=dict(fc="white", ec=col_b, pad=0.3),
        arrowprops=dict(arrowstyle="->", color=col_b, lw=2)
    )

    # Annotate WORST
    w0, w1 = worst_idx*STEP, worst_idx*STEP + WINDOW_SIZE
    mid_w = (w0 + w1)//2
    col_w = cmap(norm(worst_sim))
    ax.plot(dates[w0:w1], prices[w0:w1], ':', color=col_w, lw=2)
    ax.annotate(
        f"WORST\n{worst_sim:.3f}",
        xy=(dates[mid_w], prices[mid_w]),
        xytext=(0, -30), textcoords="offset points",
        ha="center", va="top",
        fontsize=12, fontweight="bold",
        color=col_w,
        bbox=dict(fc="white", ec=col_w, pad=0.3),
        arrowprops=dict(arrowstyle="->", color=col_w, lw=2)
    )

    # Bottom: embedding subplots
    ax1 = fig.add_subplot(gs[1,0])
    E_ref = embs[ref_idx]
    ax1.scatter(E_ref[:,0], E_ref[:,1], color='k', s=30)
    ax1.set_title(f"Window {ref_idx}", fontsize=12)
    ax1.set_xlabel("Delay-0"); ax1.set_ylabel("Delay-1")

    ax2 = fig.add_subplot(gs[1,1])
    E_best = embs[best_idx]
    ax2.scatter(E_best[:,0], E_best[:,1], c=[best_sim]*len(E_best),
                cmap=cmap, norm=norm, s=30)
    ax2.set_title(f"Best {best_idx}", fontsize=12)
    ax2.set_xlabel("Delay-0"); ax2.set_ylabel("Delay-1")

    ax3 = fig.add_subplot(gs[1,2])
    E_worst = embs[worst_idx]
    ax3.scatter(E_worst[:,0], E_worst[:,1], c=[worst_sim]*len(E_worst),
                cmap=cmap, norm=norm, s=30)
    ax3.set_title(f"Worst {worst_idx}", fontsize=12)
    ax3.set_xlabel("Delay-0"); ax3.set_ylabel("Delay-1")

    # save
    out_path = os.path.join(OUT_DIR, f"{TICKER}_ref{ref_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Main Loop
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for idx in range(n_windows):
        plot_ref_window(idx)
