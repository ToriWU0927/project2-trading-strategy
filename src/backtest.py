"""Part C backtest engine: positions, accounting, metrics, bootstrap.

Timing (Part A convention): row t = decision date. Forecast and weight w_t
are formed after the close of t, traded at the open of t+1, and earn
y_ret_t = Open(t+2)/Open(t+1) - 1. w_t pairs with y_ret_t on the SAME row;
there is no extra shift.
"""
import numpy as np
import pandas as pd

ANN = 252


# ---------------------------------------------------------------- data
def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        from mini_parquet import read_parquet
        return read_parquet(path)[0]


def calib_vol(panel, start):
    """vol_rv_20 over development rows strictly BEFORE `start` (first decision date of the period evaluated).
    Validation (from 2005) is calibrated on 1994-2004 only; the test period (from 2015) on all of 1994-2014.
    Revision R2 (2026-11-09): earlier versions used all development rows for both periods."""
    m = (panel["period"] == "dev") & (panel.index < start)
    return panel.loc[m, "vol_rv_20"]


def realised_window(panel_index, decision_dates):
    """(first execution open, last realisation open) for a set of decision dates t:
    row t trades at the open of t+1 and is realised at the open of t+2 (R7)."""
    loc = panel_index.get_indexer(decision_dates)
    return str(panel_index[loc.min() + 1].date()), str(panel_index[loc.max() + 2].date())


# ---------------------------------------------------------------- positions
def w_mv(mu, sigma2, gamma, lo=0.0, hi=1.0):
    """Mean-variance weight: argmax w*mu - gamma/2 w^2 sigma2, clipped."""
    return np.clip(np.asarray(mu) / (gamma * np.asarray(sigma2)), lo, hi)


def w_cost_aware(mu, sigma2, gamma, c, w0=0.0, lo=0.0, hi=1.0):
    """Single-period optimum of w*mu - gamma/2 w^2 sigma2 - c|w - w_prev|.

    The objective is concave; its maximiser is w_prev projected onto the
    no-trade band [(mu-c)/(gamma sigma2), (mu+c)/(gamma sigma2)], then onto
    [lo, hi]. w_prev is the previous TARGET weight (known at close t).
    """
    mu, sigma2 = np.asarray(mu, float), np.asarray(sigma2, float)
    out = np.empty(len(mu)); prev = w0
    for i in range(len(mu)):
        k = gamma * sigma2[i]
        L, U = (mu[i] - c) / k, (mu[i] + c) / k
        w = min(max(prev, L), U)
        w = min(max(w, lo), hi)
        out[i] = w; prev = w
    return out


def w_sign(mu):
    return (np.asarray(mu) > 0).astype(float)


# ---------------------------------------------------------------- accounting
def run(w, y_ret, rf, c, w0=0.0, index=None):
    """Daily ledger. Returns DataFrame with turnover, cost, gross/net returns, NAV, drawdown.

    turnover_t = |w_t - w_drift_{t-1}| where w_drift is the SPY weight after the
    previous holding period's price move (exact trade size); turnover_approx_t =
    |w_t - w_{t-1}| is the target-difference approximation.
    """
    w = np.asarray(w, float); y = np.asarray(y_ret, float); r = np.asarray(rf, float)
    assert not (np.isnan(w).any() or np.isnan(y).any() or np.isnan(r).any()), "NaN in inputs"
    gross = w * y + (1 - w) * r
    w_prev = np.r_[w0, w[:-1]]
    drift = np.empty_like(w)
    drift[0] = w0
    drift[1:] = w[:-1] * (1 + y[:-1]) / (1 + gross[:-1])
    turnover = np.abs(w - drift)
    cost = c * turnover
    net = gross - cost
    nav = np.cumprod(1 + net)
    dd = nav / np.maximum.accumulate(np.r_[1.0, nav])[1:] - 1
    return pd.DataFrame({
        "w": w, "w_prev_target": w_prev, "w_drift_before_trade": drift,
        "turnover": turnover, "turnover_approx": np.abs(w - w_prev),
        "cost": cost, "y_ret": y, "rf_daily": r,
        "r_gross": gross, "r_net": net, "r_net_ex": net - r,
        "nav": nav, "drawdown": dd}, index=index)


# ---------------------------------------------------------------- metrics
def max_dd_duration(nav):
    """Longest stretch (in trading periods) below a previous NAV peak (incl. start = 1)."""
    peak = 1.0; cur = best = 0
    for v in nav:
        if v >= peak: peak = v; cur = 0
        else: cur += 1; best = max(best, cur)
    return best


def metrics(led, gamma_ce=3, name=None):
    ex = led["r_net_ex"].values; n = len(ex)
    years = n / ANN
    nav = led["nav"].values
    mu, sd = ex.mean() * ANN, ex.std(ddof=1) * np.sqrt(ANN)
    gross_ex = (led["r_gross"] - led["rf_daily"]).values.mean() * ANN
    out = {
        "CAGR": nav[-1] ** (1 / years) - 1,
        "ann_excess": mu,
        "ann_excess_vol": sd,                       # volatility of net EXCESS return (Sharpe denominator); was "ann_vol"
        "ann_return_vol": led["r_net"].std(ddof=1) * np.sqrt(ANN),   # volatility of net return itself (R5)
        "sharpe": mu / sd if sd > 0 else np.nan,
        "max_drawdown": led["drawdown"].min(),
        "max_dd_duration_days": max_dd_duration(nav),
        "ann_turnover": led["turnover"].sum() / years,
        "avg_weight": led["w"].mean(),
        "share_days_invested": (led["w"] > 1e-9).mean(),
        "ann_cost_drag": led["cost"].sum() / years,
        "ann_excess_gross": gross_ex,
        "CE_ann": mu - gamma_ce / 2 * sd ** 2,
        "final_nav": nav[-1],
        "n_periods": n,
    }
    return pd.Series(out, name=name)


# ---------------------------------------------------------------- bootstrap
def block_indices(n, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def _stats(ex, gamma):
    mu, sd = ex.mean() * ANN, ex.std(ddof=1) * np.sqrt(ANN)
    return mu, (mu / sd if sd > 0 else np.nan), mu - gamma / 2 * sd ** 2


def paired_bootstrap(ex_a, ex_b, block=63, reps=2000, seed=5205, gamma=3):
    """Paired circular block bootstrap of differences A - B (same resampled days)."""
    ex_a, ex_b = np.asarray(ex_a), np.asarray(ex_b); n = len(ex_a)
    rng = np.random.default_rng(seed)
    pa, pb = _stats(ex_a, gamma), _stats(ex_b, gamma)
    point = {"ann_excess_diff": pa[0] - pb[0], "sharpe_diff": pa[1] - pb[1], "ce_diff": pa[2] - pb[2]}
    draws = {k: np.empty(reps) for k in point}
    for b in range(reps):
        i = block_indices(n, block, rng)
        a, bb = _stats(ex_a[i], gamma), _stats(ex_b[i], gamma)
        draws["ann_excess_diff"][b] = a[0] - bb[0]
        draws["sharpe_diff"][b] = a[1] - bb[1]
        draws["ce_diff"][b] = a[2] - bb[2]
    rows = []
    for k, v in point.items():
        d = draws[k]
        rows.append({"stat": k, "point": v, "ci_lo_95": np.nanpercentile(d, 2.5),
                     "ci_hi_95": np.nanpercentile(d, 97.5), "share_draws_le_0": np.mean(d <= 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- input guards (review round 2, N1 / N5)
VAL_YEARS = (2005, 2014)   # Part B validation folds: one calendar year each, 2005-2014


def expected_dates(panel, period):
    """Decision dates the forecasts must cover exactly: panel test rows, or dev rows in the validation years."""
    if period == "test":
        return panel.index[panel["period"] == "test"]
    yrs = panel.index.year
    return panel.index[(panel["period"] == "dev") & (yrs >= VAL_YEARS[0]) & (yrs <= VAL_YEARS[1])]


def aligned_index(panel, pred, period, cols):
    """Validate forecasts against the panel and return (decision index, dropped trailing dates).

    Refuses duplicate, unsorted, missing or unexpected dates instead of silently skipping holding periods.
    Only a trailing block without a realised label (last rows of the sample) may be dropped.
    Forecast columns, y_ret, y_exret, rf_daily must be finite and vol_rv_20 strictly positive on kept rows."""
    pi = pred.index
    if not pi.is_unique:
        raise ValueError(f"duplicate forecast dates: {sorted(set(str(d.date()) for d in pi[pi.duplicated()]))[:5]}")
    if not pi.is_monotonic_increasing:
        raise ValueError("forecast dates are not sorted in increasing order")
    exp = expected_dates(panel, period)
    missing, extra = exp.difference(pi), pi.difference(exp)
    if len(missing) or len(extra):
        raise ValueError(f"{period} forecasts do not cover the expected dates: {len(missing)} missing "
                         f"{[str(d.date()) for d in missing[:5]]}, {len(extra)} unexpected {[str(d.date()) for d in extra[:5]]}")
    lab = panel.loc[pi, "y_ret"].notna().values
    k = int(lab.sum())
    if not lab[:k].all():
        raise ValueError("a row without realised return lies inside the period; only the trailing rows may lack a label")
    idx = pi[:k]
    f = panel.loc[idx]
    for c in ["y_ret", "y_exret", "rf_daily", "vol_rv_20"]:
        if not np.isfinite(f[c].values).all():
            raise ValueError(f"non-finite {c} on a kept row")
    if not (f["vol_rv_20"].values > 0).all():
        raise ValueError("vol_rv_20 must be strictly positive (risk variance > 0)")
    for c in cols:
        if not np.isfinite(pred.loc[idx, c].values.astype(float)).all():
            raise ValueError(f"non-finite forecast in column {c}")
    return idx, [str(d.date()) for d in pi[k:]]


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def impl_sha256():
    """Fingerprint of the strategy/accounting implementation only (figure code does not enter)."""
    import hashlib, inspect
    src = "".join(inspect.getsource(fn) for fn in (calib_vol, w_mv, w_cost_aware, run, metrics, aligned_index))
    return hashlib.sha256(src.encode()).hexdigest()


def selection_mismatches(recorded, current, tol=1e-10):
    """Compare a stored validation selection with what the current code/data/config reproduce. Returns a list of problems."""
    bad = []
    for k, v in current.items():
        r = recorded.get(k, "<missing>")
        if isinstance(v, dict):
            if not isinstance(r, dict) or set(r) != set(v):
                bad.append(f"{k}: keys {sorted(r) if isinstance(r, dict) else r} vs {sorted(v)}"); continue
            for kk in v:
                a, b = r[kk], v[kk]
                same = np.allclose(a, b, rtol=0, atol=tol) if not isinstance(b, str) else a == b
                if not same: bad.append(f"{k}[{kk}]: recorded {a} vs current {b}")
        elif isinstance(v, str):
            if r != v: bad.append(f"{k}: recorded {r} vs current {v}")
        elif not (isinstance(r, (int, float)) and abs(r - v) <= tol):
            bad.append(f"{k}: recorded {r} vs current {v}")
    return bad
