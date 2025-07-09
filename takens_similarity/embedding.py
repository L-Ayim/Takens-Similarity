import numpy as np

def takens_embedding(series: np.ndarray, delay: int = 1, dim: int = 2) -> np.ndarray:
    series = np.asarray(series)
    N = len(series) - (dim - 1) * delay
    if N <= 0:
        raise ValueError("Series too short for delay/dim")
    emb = np.empty((N, dim))
    for i in range(dim):
        emb[:, i] = series[i*delay : i*delay + N]
    return emb

def sliding_window_embeddings(series: np.ndarray, window_size: int, step: int,
                              delay: int = 1, dim: int = 2) -> list[np.ndarray]:
    series = np.asarray(series)
    embs = []
    for start in range(0, len(series) - window_size + 1, step):
        win = series[start:start+window_size]
        embs.append(takens_embedding(win, delay=delay, dim=dim))
    return embs
