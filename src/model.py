"""Part B -- model & validation (member B).

Implements the pre-declared protocol:

  label       : ``y_exret``  (open-to-open next-day SPY excess return)
  features    : M1 / M2 / M3 groups from ``feature_groups.json`` (unscaled)
  scaling     : StandardScaler fit on the train fold only, then clip to +/- clip_sigma
  baselines   : historical (expanding) mean ; AR(1) via the ``ret_cc`` lag
  models      : Ridge / Lasso / ElasticNet with small pre-declared grids
  validation  : expanding window on the dev period, one year per fold (2005-2014),
                with an embargo of ``embargo_days`` rows before each validation year
  selection   : per (family, feature set) pick the hyperparameter maximising the mean
                validation out-of-sample R2 ; the champion is the argmax over all
                (family, feature set) pairs
  test        : freeze hyperparameters, then monthly walk-forward refit of coefficients

Out-of-sample R2 uses the *expanding train mean* as the forecast baseline
(Campbell-Thompson style), so it is a true out-of-sample number.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]  # repo root


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_config(cfg_path=None) -> dict:
    if cfg_path is None:
        cfg_path = ROOT / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)["partB"]


def load_splits(cfg: dict) -> dict:
    with open(ROOT / cfg["paths"]["splits_config"], "r", encoding="utf-8") as f:
        return json.load(f)["splits"]


def load_panel(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(ROOT / cfg["paths"]["panel"])


def load_groups(cfg: dict) -> dict:
    with open(ROOT / cfg["paths"]["feature_groups"], "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
class ClipScaler(BaseEstimator, TransformerMixin):
    """Standardize (fit on train only) then clip to ``[-clip, clip]``."""

    def __init__(self, clip: float = 5.0):
        self.clip = clip

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        Z = self.scaler_.transform(X)
        return np.clip(Z, -self.clip, self.clip)

    def get_feature_names_out(self, input_features=None):
        return input_features


class RFF(BaseEstimator, TransformerMixin):
    """Random Fourier features (Rahimi & Recht 2007), approximating an RBF kernel.

    ``gamma = 1 / sigma^2``. The random weights are drawn once per fit with a fixed
    seed, so they are data-independent (not leakage) and reproducible across folds.
    """

    def __init__(self, n_components: int = 100, gamma: float = 1.0, random_state: int = 5205):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        self.W_ = rng.normal(0.0, np.sqrt(self.gamma), size=(d, self.n_components))
        self.b_ = rng.uniform(0.0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        Z = X @ self.W_ + self.b_
        return np.sqrt(2.0 / self.n_components) * np.cos(Z)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"rff_{i}" for i in range(self.n_components)], dtype=object)


def expand_grid(spec: dict) -> list[dict]:
    keys = list(spec.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(spec[k] for k in keys))]


def make_pipeline(family: str, params: dict, clip: float) -> Pipeline:
    steps = [("scale", ClipScaler(clip=clip))]
    if family == "ridge":
        m = Ridge(alpha=params["alpha"])
    elif family == "lasso":
        m = Lasso(alpha=params["alpha"], max_iter=20000, tol=1e-4)
    elif family == "elasticnet":
        m = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                       max_iter=20000, tol=1e-4)
    else:
        raise ValueError(f"unknown family: {family}")
    steps.append(("model", m))
    return Pipeline(steps)


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray, baseline_mean: float) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - baseline_mean) ** 2))
    return 1.0 - ss_res / ss_tot


def expanding_folds(dev: pd.DataFrame, val_years: list[int], embargo: int):
    """Yield (year, train, val). Train ends ``embargo`` rows before the val year.

    The ``embargo`` rows are the last rows whose label still uses opens inside the
    validation year, so they must not enter the training set.
    """
    for yr in val_years:
        val = dev[dev.index.year == yr]
        train = dev[dev.index < val.index[0]].iloc[:-embargo]
        yield yr, train, val


# --------------------------------------------------------------------------- #
# Cross-validation / selection
# --------------------------------------------------------------------------- #
def run_cv(dev: pd.DataFrame, groups: dict, families: list[str], feature_sets: list[str],
           grids: dict, val_years: list[int], embargo: int, label: str, clip: float,
           winsor: tuple | None = None) -> pd.DataFrame:
    """One row per (family, feature set, hyperparameter): mean validation OOS R2.

    If ``winsor = (lo, hi)`` is given, the label is clipped to the [lo, hi] quantiles
    computed from the *train fold only* (leakage-free), before fitting and scoring.
    """
    rows = []
    for fam in families:
        for fs in feature_sets:
            feats = groups[fs]
            for params in expand_grid(grids[fam]):
                fold_r2 = []
                for _yr, train, val in expanding_folds(dev, val_years, embargo):
                    Xtr = train[feats].to_numpy()
                    Xva = val[feats].to_numpy()
                    ytr = train[label].to_numpy()
                    yva = val[label].to_numpy()
                    if winsor is not None:
                        lo, hi = winsor
                        qlo, qhi = np.quantile(ytr, lo), np.quantile(ytr, hi)
                        ytr = np.clip(ytr, qlo, qhi)
                        yva = np.clip(yva, qlo, qhi)
                    pipe = make_pipeline(fam, params, clip)
                    pipe.fit(Xtr, ytr)
                    fold_r2.append(oos_r2(yva, pipe.predict(Xva), float(ytr.mean())))
                rows.append({"family": fam, "feature_set": fs, **params,
                             "mean_oos_r2": float(np.mean(fold_r2))})
    return pd.DataFrame(rows)


def select_best(cv_df: pd.DataFrame):
    """Return (selected_df, champion). Selected = best params per (family, feature set)."""
    idx = cv_df.groupby(["family", "feature_set"])["mean_oos_r2"].idxmax()
    selected = cv_df.loc[idx].reset_index(drop=True)
    champion = selected.loc[selected["mean_oos_r2"].idxmax()]
    return selected, champion


def run_rff_cv(dev: pd.DataFrame, feats: list[str], n_components: list[int], alphas: list[float],
               val_years: list[int], embargo: int, label: str, clip: float,
               gamma: float, seed: int) -> pd.DataFrame:
    """Kelly-style complexity exploration: RFF(n_components) + Ridge(alpha), on one feature set."""
    rows = []
    for k in n_components:
        for a in alphas:
            fold_r2 = []
            for _yr, train, val in expanding_folds(dev, val_years, embargo):
                pipe = Pipeline([
                    ("scale", ClipScaler(clip=clip)),
                    ("rff", RFF(n_components=k, gamma=gamma, random_state=seed)),
                    ("ridge", Ridge(alpha=a)),
                ])
                pipe.fit(train[feats].to_numpy(), train[label].to_numpy())
                fold_r2.append(oos_r2(val[label].to_numpy(),
                                      pipe.predict(val[feats].to_numpy()),
                                      float(train[label].mean())))
            rows.append({"n_components": k, "alpha": a,
                         "mean_oos_r2": float(np.mean(fold_r2))})
    return pd.DataFrame(rows)


def baseline_cv(dev: pd.DataFrame, label: str, ar_feature: str,
                val_years: list[int], embargo: int) -> float:
    """Mean validation OOS R2 of the AR(1) baseline (ret_cc lag)."""
    ar_r2 = []
    for _yr, train, val in expanding_folds(dev, val_years, embargo):
        ytr, yva = train[label].to_numpy(), val[label].to_numpy()
        m = LinearRegression().fit(train[[ar_feature]].to_numpy(), ytr)
        ar_r2.append(oos_r2(yva, m.predict(val[[ar_feature]].to_numpy()), float(ytr.mean())))
    return float(np.mean(ar_r2))


# --------------------------------------------------------------------------- #
# Walk-forward on the test period (hyperparameters frozen)
# --------------------------------------------------------------------------- #
def _months(panel: pd.DataFrame, label: str, embargo: int):
    """Yield (month_df, train_df). Refit at each month's first trading day."""
    usable = panel[panel.period != "warmup"]
    test = panel[panel.period == "test"]
    for _p, month in test.groupby(test.index.to_period("M")):
        d = month.index[0]
        i = usable.index.get_loc(d)
        train = usable.iloc[: i - embargo].dropna(subset=[label])
        yield month, train


def model_walkforward(panel, feats, family, params, label, embargo, clip):
    """Monthly walk-forward. Returns (test predictions, {month_start: coef_array})."""
    test = panel[panel.period == "test"]
    preds = pd.Series(index=test.index, dtype=float)
    coefs = {}
    for month, train in _months(panel, label, embargo):
        pipe = make_pipeline(family, params, clip)
        pipe.fit(train[feats].to_numpy(), train[label].to_numpy())
        preds.loc[month.index] = pipe.predict(month[feats].to_numpy())
        coefs[month.index[0]] = pipe.named_steps["model"].coef_.copy()
    return preds, coefs


def mean_walkforward(panel, label, embargo) -> pd.Series:
    """Expanding historical-mean forecast (the naive 'no predictability' baseline)."""
    test = panel[panel.period == "test"]
    preds = pd.Series(index=test.index, dtype=float)
    for month, train in _months(panel, label, embargo):
        preds.loc[month.index] = float(train[label].mean())
    return preds


def ar1_walkforward(panel, label, ar_feature, embargo) -> pd.Series:
    test = panel[panel.period == "test"]
    preds = pd.Series(index=test.index, dtype=float)
    for month, train in _months(panel, label, embargo):
        m = LinearRegression().fit(train[[ar_feature]].to_numpy(), train[label].to_numpy())
        preds.loc[month.index] = m.predict(month[[ar_feature]].to_numpy())
    return preds


def val_predictions(dev, feats, family, params, label, val_years, embargo, clip) -> pd.DataFrame:
    """Concatenated out-of-sample validation predictions for a single config."""
    rows = []
    for _yr, train, val in expanding_folds(dev, val_years, embargo):
        pipe = make_pipeline(family, params, clip)
        pipe.fit(train[feats].to_numpy(), train[label].to_numpy())
        p = pipe.predict(val[feats].to_numpy())
        tm = float(train[label].mean())
        for d, yt, yh in zip(val.index, val[label].to_numpy(), p):
            rows.append({"date": d, "y_true": yt, "y_hat": yh, "train_mean": tm})
    return pd.DataFrame(rows).set_index("date")
