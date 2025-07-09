
import numpy as np
from scipy.linalg import svd

def _procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the minimal root-mean-square deviation D between X and Y
    under optimal translation, rotation and uniform scaling.
    """
    # center clouds
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0, Y0 = X - muX, Y - muY

    # optimal scale & rotation come from SVD of X0^T Y0
    U, s, Vt = svd(X0.T @ Y0)
    R = U @ Vt                # optimal rotation
    # optimal scale c = trace(S) / ||X0||^2
    c = np.sum(s) / np.sum(X0**2)

    # apply transform
    X_aligned = c * X0 @ R

    # translation b = muY - c*muX*R
    b = muY - c * muX @ R

    # RMS deviation
    diff = X_aligned + b - Y
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def compute_similarities_to_ref(
    embs: list[np.ndarray],
    ref_idx: int
) -> np.ndarray:
    """
    For each embedding E in `embs`, compute the Procrustes-based RMS
    distance D to the reference embedding at `ref_idx`, then convert
    to a similarity score 1/(1 + D).
    """
    if not (0 <= ref_idx < len(embs)):
        raise IndexError("ref_idx out of range")

    E_ref = embs[ref_idx]
    ds = np.array([_procrustes_distance(E, E_ref) for E in embs])
    # similarity in (0,1]
    sims = 1.0 / (1.0 + ds)
    return sims


def find_best_and_worst(
    sims: np.ndarray,
    ref_idx: int
) -> tuple[int, float, int, float]:
    sims = np.asarray(sims, float)
    if not (0 <= ref_idx < sims.size):
        raise IndexError("ref_idx out of range")

    tmp = sims.copy()
    tmp[ref_idx] = -np.inf
    best_idx, best_sim = int(np.argmax(tmp)), float(np.max(tmp))
    tmp[ref_idx] = np.inf
    worst_idx, worst_sim = int(np.argmin(tmp)), float(np.min(tmp))
    return best_idx, best_sim, worst_idx, worst_sim
