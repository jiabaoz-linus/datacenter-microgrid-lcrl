"""
Bootstrap intervals for methods evaluated on aligned test days.

The default crossed scheme resamples training seeds separately for each
method, then applies one set of resampled test-day indices to all rows and
methods. Each day-level value retains its aggregated equipment samples.
Single-row methods receive only test-day resampling.

The optional nested scheme uses a separate test-day index vector for each
resampled row. Corresponding row index vectors are shared across methods.
"""
from typing import Dict
import numpy as np


def _replicate(v, s_idx, d_idx, scheme):
    if v.shape[0] == 1:                       # Single-row input: resample days only.
        d = d_idx if d_idx.ndim == 1 else d_idx[0]
        return v[0, d].mean()
    if scheme == "crossed":
        return v[s_idx][:, d_idx].mean()
    return v[s_idx[:, None], d_idx[: len(s_idx)]].mean()


def hierarchical_bootstrap(values: Dict[str, np.ndarray], n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0, scheme: str = "crossed"):
    """Compute percentile intervals for means and paired method differences.

    values[method] has shape (n_seeds_m, n_days), with aligned day columns
    across methods. The number of seed rows may differ between methods.
    Return two dictionaries of (mean, lower, upper): method means and
    differences between the first method and each remaining method."""
    assert scheme in ("crossed", "nested")
    rng = np.random.default_rng(seed)
    names = list(values)
    Dn = values[names[0]].shape[1]
    for m in names:
        assert values[m].shape[1] == Dn, f"{m}: test days not aligned"
        if values[m].shape[0] > 1:
            assert not all(np.array_equal(values[m][0], r) for r in values[m][1:]), \
                f"{m}: all seed rows are identical - pass a single row instead of copies"
    S_max = max(values[m].shape[0] for m in names)
    boots = {m: np.zeros(n_boot) for m in names}
    for b in range(n_boot):
        d_idx = rng.integers(0, Dn, Dn) if scheme == "crossed" else rng.integers(0, Dn, (S_max, Dn))
        for m in names:
            S = values[m].shape[0]
            boots[m][b] = _replicate(values[m], rng.integers(0, S, S), d_idx, scheme)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out = {m: (float(values[m].mean()), *np.percentile(boots[m], q)) for m in names}
    ref = names[0]
    diff = {m: (float(values[ref].mean() - values[m].mean()), *np.percentile(boots[ref] - boots[m], q))
            for m in names[1:]}
    return out, diff
