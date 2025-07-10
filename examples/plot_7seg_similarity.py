#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

from takens_similarity.embedding import takens_embedding
from takens_similarity.similarity import compute_similarities_to_ref

# ───────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ───────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots", "7seg_synthetic")
os.makedirs(OUT_DIR, exist_ok=True)

REF_IDX = 0    # “0” is our reference digit
DELAY   = 1
DIM     = 3

# Segment geometry parameters
SEG_LENGTH = 1.0
SEG_W      = 0.2

# Each segment as a 4-corner polygon in (x,y)
SEG_POLYGONS = {
    'A': [(SEG_W, 0), (SEG_W+SEG_LENGTH, 0),
          (SEG_W+SEG_LENGTH-SEG_W, SEG_W), (SEG_W+SEG_W, SEG_W)],
    'B': [(SEG_W+SEG_LENGTH, SEG_W),
          (SEG_W+SEG_LENGTH, SEG_W+SEG_LENGTH),
          (SEG_W+SEG_LENGTH-SEG_W, SEG_W+SEG_LENGTH-SEG_W),
          (SEG_W+SEG_LENGTH-SEG_W, SEG_W)],
    'C': [(SEG_W+SEG_LENGTH, SEG_W+SEG_LENGTH+SEG_W),
          (SEG_W+SEG_LENGTH, 2*SEG_LENGTH+SEG_W),
          (SEG_W+SEG_LENGTH-SEG_W, 2*SEG_LENGTH),
          (SEG_W+SEG_LENGTH-SEG_W, SEG_W+SEG_LENGTH+SEG_W)],
    'D': [(SEG_W, 2*SEG_LENGTH+SEG_W),
          (SEG_W+SEG_LENGTH, 2*SEG_LENGTH+SEG_W),
          (SEG_W+SEG_LENGTH-SEG_W, 2*SEG_LENGTH),
          (SEG_W+SEG_W, 2*SEG_LENGTH)],
    'E': [(0, SEG_W+SEG_LENGTH+SEG_W),
          (0, 2*SEG_LENGTH+SEG_W),
          (SEG_W, 2*SEG_LENGTH),
          (SEG_W, SEG_W+SEG_LENGTH+SEG_W)],
    'F': [(0, SEG_W),
          (0, SEG_W+SEG_LENGTH),
          (SEG_W, SEG_W+SEG_LENGTH-SEG_W),
          (SEG_W, SEG_W)],
    'G': [(SEG_W, SEG_LENGTH),
          (SEG_W+SEG_LENGTH, SEG_LENGTH),
          (SEG_W+SEG_LENGTH-SEG_W, SEG_LENGTH+SEG_W),
          (SEG_W+SEG_W, SEG_LENGTH+SEG_W)]
}

# Which segments lit for digits “0”–“3”
DIGITS = {
    '0': ['A','B','C','D','E','F'],
    '1': ['B','C'],
    '2': ['A','B','G','E','D'],
    '3': ['A','B','C','D','G']
}

# ───────────────────────────────────────────────────────────────
# 1. Build fixed‐length landmark arrays
# ───────────────────────────────────────────────────────────────
lms    = []
labels = []
for digit, segs in DIGITS.items():
    pts = []
    # Always iterate all 7 segments in the same order
    for seg_name in ['A','B','C','D','E','F','G']:
        poly = SEG_POLYGONS[seg_name]
        if seg_name in segs:
            # lit segment: include its 4 corners
            pts.extend(poly)
        else:
            # unlit: repeat first corner 4× (degenerate)
            pts.extend([poly[0]] * 4)
    lms.append(np.array(pts, float))
    labels.append(digit)

# Add a duplicate “0” as control
lms.append(lms[REF_IDX].copy())
labels.append(f"{labels[REF_IDX]} (ctrl)")

n = len(lms)

# ───────────────────────────────────────────────────────────────
# 2. Flatten & normalize
# ───────────────────────────────────────────────────────────────
signals    = [lm.flatten() for lm in lms]            # each length 56
signals_nz = [(s - s.mean())/s.std() for s in signals]

# ───────────────────────────────────────────────────────────────
# 3. Takens embeddings & similarity
# ───────────────────────────────────────────────────────────────
embs = [
    takens_embedding(sig, delay=DELAY, dim=DIM)
    for sig in signals_nz
]
sims = compute_similarities_to_ref(embs, REF_IDX)

# ───────────────────────────────────────────────────────────────
# 4. Plot 3×n grid
# ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(3*n, 8))
gs  = GridSpec(3, n, figure=fig, hspace=0.4, wspace=0.3)

for i in range(n):
    color = 'black' if i == REF_IDX else plt.cm.plasma(sims[i])

    # Row 0: synthetic 7-seg display
    ax0 = fig.add_subplot(gs[0, i])
    # draw each polygon from its corners
    poly_pts = lms[i].reshape(-1, 4, 2)
    for quad in poly_pts:
        patch = Polygon(quad, closed=True, facecolor='red', edgecolor='k', alpha=0.8)
        ax0.add_patch(patch)
    ax0.set_xlim(-0.2, SEG_LENGTH+0.2)
    ax0.set_ylim(2*SEG_LENGTH+SEG_W+0.2, -0.2)
    ax0.set_aspect('equal')
    ax0.axis('off')
    title = labels[i] if i==REF_IDX else f"{labels[i]}\nsim={sims[i]:.2f}"
    ax0.set_title(title, fontsize=10)

    # Row 1: 1D normalized signal
    ax1 = fig.add_subplot(gs[1, i])
    ax1.plot(signals_nz[i], color=color, linewidth=1)
    ax1.set_xlim(0, len(signals_nz[i]))
    ax1.set_ylabel("norm", fontsize=8)
    ax1.set_title("flat signal", fontsize=8)
    ax1.tick_params(labelsize=6)

    # Row 2: 2D Takens embedding
    ax2 = fig.add_subplot(gs[2, i])
    E = embs[i]
    ax2.scatter(E[:,0], E[:,1], c=color, s=20)
    ax2.set_aspect('equal')
    ax2.set_xlabel("Delay-0", fontsize=8)
    ax2.set_ylabel("Delay-1", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=6)

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────
# 5. Save figure
# ───────────────────────────────────────────────────────────────
out = os.path.join(OUT_DIR, "7seg_synthetic_similarity.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved →", out)
