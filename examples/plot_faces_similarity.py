#!/usr/bin/env python3
import os
import numpy as np
from skimage import io
import face_alignment
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from takens_similarity.embedding import takens_embedding
from takens_similarity.similarity import compute_similarities_to_ref

# ──────────────────────────────────────────────────────────────────────────────
# 0. Compute base directory (script’s own folder), and set up paths relative to it
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(BASE_DIR, "images", "faces")
OUT_DIR  = os.path.join(BASE_DIR, "plots", "faces_full_pipeline_control")
os.makedirs(OUT_DIR, exist_ok=True)

REF_IDX = 0    # index of reference face
DELAY   = 1
DIM     = 3

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load face‐alignment model, images & landmarks
# ──────────────────────────────────────────────────────────────────────────────
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device="cpu"
)

images, lms, labels = [], [], []
for fname in sorted(f for f in os.listdir(IMG_DIR)
                    if f.lower().endswith((".jpg","jpeg","png"))):
    full_path = os.path.join(IMG_DIR, fname)
    img = io.imread(full_path)
    preds = fa.get_landmarks(img)
    if not preds:
        continue
    images.append(img)
    lms.append(preds[0].astype(float))
    labels.append(fname)

# duplicate reference at end for control
images.append(images[REF_IDX])
lms.append(lms[REF_IDX])
labels.append(f"{labels[REF_IDX]} (control)")

n = len(images)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Flatten & normalize landmark signals
# ──────────────────────────────────────────────────────────────────────────────
signals    = [lm.flatten() for lm in lms]
signals_nz = [(s - s.mean())/s.std() for s in signals]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Build Takens embeddings & compute similarities
# ──────────────────────────────────────────────────────────────────────────────
embs = [takens_embedding(sig, delay=DELAY, dim=DIM)
        for sig in signals_nz]
sims = compute_similarities_to_ref(embs, REF_IDX)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Plot everything in a 4×n grid
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(3*n, 10))
gs  = GridSpec(4, n, figure=fig, hspace=0.4, wspace=0.2)

for i in range(n):
    img   = images[i]
    H, W  = img.shape[:2]
    color = "black" if i == REF_IDX else plt.cm.plasma(sims[i])

    # Row 0: photo
    ax0 = fig.add_subplot(gs[0, i])
    ax0.imshow(img)
    title = labels[i] if i == REF_IDX else f"{labels[i]}\nsim={sims[i]:.2f}"
    ax0.set_title(title, fontsize=9)
    ax0.axis("off")

    # Row 1: landmarks overlay
    ax1 = fig.add_subplot(gs[1, i])
    ax1.scatter(lms[i][:,0], lms[i][:,1], c=color, s=12)
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("landmarks", fontsize=8)

    # Row 2: normalized flattened signal
    ax2 = fig.add_subplot(gs[2, i])
    ax2.plot(signals_nz[i], color=color, linewidth=1)
    ax2.set_xlim(0, len(signals_nz[i]))
    ax2.set_ylabel("norm", fontsize=8)
    ax2.set_title("signal", fontsize=8)
    ax2.tick_params(labelsize=6)

    # Row 3: Takens embedding (2D)
    ax3 = fig.add_subplot(gs[3, i])
    E = embs[i]
    ax3.scatter(E[:,0], E[:,1], c=color, s=15)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Delay-0", fontsize=8)
    ax3.set_ylabel("Delay-1", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=6)

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Save to relative OUT_DIR
# ──────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "faces_full_pipeline_control.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("Saved →", out_path)
