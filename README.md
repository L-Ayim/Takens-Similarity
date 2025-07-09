# takens-similarity

[![PyPI version](https://img.shields.io/pypi/v/takens-similarity.svg)](https://pypi.org/project/takens-similarity/)  
[![License](https://img.shields.io/pypi/l/takens-similarity.svg)](https://github.com/L-Ayim/Takens-Similarity/blob/main/LICENSE)

Core Takens embeddings and similarity tools for time-series analysis in Python.

---

## Installation

```bash
pip install takens-similarity
```

## Quickstart

1. Compute a Takens embedding

    ```python
    import numpy as np
    from takens_similarity import takens_embedding

    # a 1D time series
    series = np.sin(np.linspace(0, 10, 100))
    # embed into 2D with delay=1
    E = takens_embedding(series, delay=1, dim=2)
    ```

2. Sliding-window embeddings

    ```python
    from takens_similarity import sliding_window_embeddings

    # split into windows of length 50, step 10
    windows = sliding_window_embeddings(series, window_size=50, step=10, delay=1, dim=2)
    ```

3. Compute similarity scores

    ```python
    from takens_similarity import compute_similarities_to_ref, find_best_and_worst

    # pick window 0 as reference
    sims = compute_similarities_to_ref(windows, ref_idx=0)
    best_idx, best_sim, worst_idx, worst_sim = find_best_and_worst(sims, ref_idx=0)

    print(f"Best match: window {best_idx} (sim={best_sim:.2f})")
    print(f"Worst match: window {worst_idx} (sim={worst_sim:.2f})")
    ```

---

## API Reference

### `takens_embedding(series, delay=1, dim=2) → ndarray`

Build the Takens-delay embedding of a 1D array.

- **series**: 1D array of floats  
- **delay**: integer delay between coordinates  
- **dim**: embedding dimension  
- **returns**: array of shape `(N - (dim-1)*delay, dim)`

### `sliding_window_embeddings(series, window_size, step, delay=1, dim=2) → List[ndarray]`

Generate Takens embeddings on sliding windows.

- **series**: 1D array  
- **window_size**: length of each window  
- **step**: shift between windows  
- **delay**, **dim**: as above  
- **returns**: list of embeddings

### `compute_similarities_to_ref(embs, ref_idx) → ndarray`

Compute normalized similarity scores (0–1) of each embedding to a reference via Procrustes (or Euclidean).

- **embs**: list of `ndarray` embeddings  
- **ref_idx**: index of the reference embedding  
- **returns**: 1D array of similarity scores

### `find_best_and_worst(sims, ref_idx) → (best_idx, best_sim, worst_idx, worst_sim)`

Pick the most- and least-similar indices (excluding the reference itself).

- **sims**: array of similarity scores  
- **ref_idx**: index to exclude  
- **returns**: tuple `(best_idx, best_sim, worst_idx, worst_sim)`

---

## Examples

- Stock data (with `yfinance`):  
  `examples/plot_takens_similarity.py`

- Synthetic functions:  
  `examples/plot_function_similarity.py`

Plots are saved under `examples/plots/…` for each demo.

---

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/…`)  
3. Commit your changes (`git commit -m "Add …"`)  
4. Push to your branch (`git push origin feature/…`)  
5. Open a Pull Request

Please adhere to PEP8 style and include tests under `tests/`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
