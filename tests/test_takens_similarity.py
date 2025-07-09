import numpy as np
import pytest
from takens_similarity import (
    takens_embedding,
    sliding_window_embeddings,
    compute_similarities_to_ref,
    find_best_and_worst,
)

def test_pipeline_on_arange():
    # --- synthetic series: a simple ramp ---
    series = np.arange(100, dtype=float)

    # parameters
    window_size = 10
    step        = 10
    delay, dim  = 1, 2

    # 1) sliding windows
    embs = sliding_window_embeddings(series, window_size, step, delay, dim)
    # expected number of windows
    expected_n = (len(series) - window_size) // step + 1
    assert len(embs) == expected_n

    # 2) first embedding matches direct call
    ref_emb = takens_embedding(series[:window_size], delay, dim)
    assert np.allclose(embs[0], ref_emb)

    # 3) compute similarities
    sims = compute_similarities_to_ref(embs, ref_idx=0)
    # shape & range
    assert sims.shape == (expected_n,)
    assert sims.min() >= 0.0 and sims.max() <= 1.0
    # reference must be most similar to itself
    assert pytest.approx(sims[0], rel=1e-6) == 1.0

    # 4) find best & worst (excluding ref)
    best_idx, best_sim, worst_idx, worst_sim = find_best_and_worst(sims, ref_idx=0)
    assert isinstance(best_idx, int) and isinstance(worst_idx, int)
    assert 0 < best_idx < expected_n
    assert 0 < worst_idx < expected_n
    assert 0.0 <= best_sim <= 1.0
    assert 0.0 <= worst_sim <= 1.0
