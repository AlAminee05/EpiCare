"""Evaluation (guidelines §4.6 and §7), self-contained so it runs without the sefef package.

* Time-series cross-validation, expanding window, SeFEF style: initial training = first third
  of the recording, test blocks = half of it, at least 3 events in every train and test set.
* Metrics: AUC of sensitivity vs time in warning, Brier score and Brier *skill* score against
  the training-period climatology, reliability table, sensitivity / time in warning / false
  alarms per day at a threshold chosen on the training period only.
* Hourly forecasts are scored on every worn, non-post-ictal hour; daily forecasts on the
  forecast issued at local midnight (non-overlapping days, like SeFEF's clock-time origin),
  whether or not the watch is worn at that hour. Post-ictal hours (< 1 h after any seizure) are
  left out of both training and scoring: they are cluster seizures, not new events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .dataset import PatientData
from .forecaster import PatientForecaster

_trapz = getattr(np, "trapezoid", None) or np.trapz
MODELS = ["base", "cycle", "wear", "prior_shift", "epicare"]


# ----------------------------------------------------------------------------- metrics
def auc_sen_tiw(p: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Area under sensitivity vs time-in-warning; 0.5 = chance, 1 = perfect."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    if y.sum() == 0 or len(y) == 0:
        return np.nan, np.array([]), np.array([])
    thr = np.unique(p)[::-1]
    sen = np.array([(p[y == 1] >= t).mean() for t in thr])
    tiw = np.array([(p >= t).mean() for t in thr])
    tiw = np.r_[0.0, tiw, 1.0]
    sen = np.r_[0.0, sen, 1.0]
    return float(_trapz(sen, tiw)), tiw, sen


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_skill(p, y, ref_p: float) -> float:
    bs_ref = brier(np.full(len(y), ref_p), y)
    return float(1 - brier(p, y) / bs_ref) if bs_ref > 0 else np.nan


def reliability_table(p, y, n_bins: int = 5) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return pd.DataFrame({"p_mean": [p.mean()], "observed": [y.mean()], "n": [len(p)]})
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = [{"p_mean": p[b == i].mean(), "observed": y[b == i].mean(), "n": int((b == i).sum())}
            for i in range(len(edges) - 1) if (b == i).any()]
    return pd.DataFrame(rows)


def false_alarms_per_day(p, y, thr: float, days: float) -> tuple[float, float, float]:
    """Warnings = runs of consecutive samples >= thr; a warning with no positive sample is false.
    Returns (sensitivity, time in warning, false alarms per day)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    on = p >= thr
    sen = float(on[y == 1].mean()) if y.sum() else np.nan
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    ends = np.flatnonzero(on & ~np.r_[on[1:], False])
    false = sum(1 for a, b in zip(starts, ends) if y[a: b + 1].sum() == 0)
    return sen, float(on.mean()), float(false / max(days, 1e-9))


# ----------------------------------------------------------------------------- splits
def tscv_splits(d: PatientData, cfg: Config | None = None) -> list[tuple[float, float]]:
    """Expanding-window folds as (train_end, test_end); the test block is [train_end, test_end)."""
    cfg = cfg or Config()
    e = cfg.eval
    t0, t1 = d.span()
    ev = d.events[(d.events >= t0) & (d.events < t1)]
    if len(ev) < e.n_min_events_train + e.n_min_events_test:
        return []
    train_end = t0 + e.initial_train_frac * (t1 - t0)
    if (ev < train_end).sum() < e.n_min_events_train:
        train_end = ev[e.n_min_events_train - 1] + 3600
    test_len = e.test_frac_of_initial * (train_end - t0)
    folds = []
    while train_end < t1:
        test_end = min(train_end + test_len, t1)
        while ((ev >= train_end) & (ev < test_end)).sum() < e.n_min_events_test and test_end < t1:
            test_end = min(test_end + 7 * 86400, t1)
        if ((ev >= train_end) & (ev < test_end)).sum() < e.n_min_events_test:
            break
        if t1 - test_end < test_len / 2:          # do not leave a tiny last block
            remaining = ((ev >= test_end) & (ev < t1)).sum()
            if remaining < e.n_min_events_test:
                test_end = t1
        folds.append((float(train_end), float(test_end)))
        train_end = test_end
    return folds


def test_index(d: PatientData, a: float, b: float, horizon_h: int, cfg: Config) -> np.ndarray:
    ok = (d.hours >= a) & (d.hours + horizon_h * 3600 <= b) & ~d.post_ictal
    if horizon_h >= 24:
        # one non-overlapping daily forecast; the watch need not be worn at that exact hour
        # (the cycle model needs no signal, the wearable part uses the last 24 h block)
        ok &= d.hod == cfg.eval.daily_issue_hour_local
    else:
        ok &= d.worn
    return np.flatnonzero(ok)


# ----------------------------------------------------------------------------- scoring
def score_fold(f: PatientForecaster, d: PatientData, train_end: float, test_end: float,
               X=None, shared_probs=None, cfg: Config | None = None) -> list[dict]:
    cfg = cfg or f.cfg
    rows = []
    for h in cfg.label.horizons_h:
        tr = test_index(d, f.t_start, train_end, h, cfg)
        te = test_index(d, train_end, test_end, h, cfg)
        if len(te) == 0:
            continue
        c_tr = f.predict_components(d, tr, X, shared_probs)[h] if len(tr) else {}
        c_te = f.predict_components(d, te, X, shared_probs)[h]
        y = d.y[h][te]
        days = len(te) / 24 if h < 24 else len(te)
        for name, p in c_te.items():
            thr = float(np.quantile(c_tr[name], 0.8)) if name in c_tr and len(c_tr[name]) else float(np.median(p))
            auc, _, _ = auc_sen_tiw(p, y)
            sen, tiw, fa = false_alarms_per_day(p, y, thr, days)
            rows.append({
                "patient": d.pid, "horizon_h": h, "model": name,
                "train_end": pd.to_datetime(train_end, unit="s", utc=True).date(),
                "test_end": pd.to_datetime(test_end, unit="s", utc=True).date(),
                "n_test": len(te), "n_pos": int(y.sum()), "AUC_TiW": auc,
                "BS": brier(p, y), "BSS": brier_skill(p, y, f.models[h].base_p),
                "thr": thr, "Sen@thr": sen, "TiW@thr": tiw, "FA_per_day": fa,
                "mode": f.models[h].mode,
            })
    return rows


def evaluate_patient(d: PatientData, cfg: Config | None = None, representation: str = "features",
                     shared_embed_fn=None, verbose: bool = True) -> tuple[pd.DataFrame, list]:
    """Run the TSCV for one patient.

    representation:
        "features" -> SeFEF hourly features (the normaliser is refitted on each training part)
        "shared"   -> ``shared_embed_fn(d) -> (embeddings, probs)``; the shared network must
                      have been trained on data that excludes this patient's test periods
                      (use leave-one-patient-out, or train_end per patient = first fold's end).
        "none"     -> cycle model and base rate only
    Returns (per-fold metrics, list of (fold, PatientForecaster, predictions DataFrame)).
    """
    cfg = cfg or Config()
    folds = tscv_splits(d, cfg)
    if verbose:
        print(f"[{d.pid}] {len(folds)} folds")
    rows, fitted = [], []
    E = P = None
    if representation == "shared":
        E, P = shared_embed_fn(d)
    for k, (tr_end, te_end) in enumerate(folds):
        X = None
        if representation == "features":
            d.fit_normalizer(tr_end)
            X, names = d.hourly_features()
        elif representation == "shared":
            X, names = E, [f"emb{j}" for j in range(E.shape[1])]
        else:
            names = None
        f = PatientForecaster(cfg, feature_names=names).fit(d, tr_end, X, P)
        r = score_fold(f, d, tr_end, te_end, X, P, cfg)
        for row in r:
            row["fold"] = k
        rows += r
        te = np.flatnonzero((d.hours >= tr_end) & (d.hours < te_end))
        fitted.append((k, f, f.predict(d, te, X, P)))
        if verbose:
            ep = [x for x in r if x["model"] == "epicare"]
            msg = ", ".join(f"{x['horizon_h']}h AUC {x['AUC_TiW']:.2f} BSS {x['BSS']:+.3f}" for x in ep)
            print(f"  fold {k}: train events {f.n_train_events}, {msg}")
    return pd.DataFrame(rows), fitted


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean over folds per patient, then mean +- sd over patients, per horizon and model."""
    if results.empty:
        return results
    per_patient = results.groupby(["horizon_h", "model", "patient"])[["AUC_TiW", "BSS", "FA_per_day"]].mean()
    agg = per_patient.groupby(["horizon_h", "model"]).agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.round(3)


def events_seen_curve(d: PatientData, cfg: Config | None = None, grid=None, test_days: float = 60,
                      representation: str = "features", shared_embed_fn=None) -> pd.DataFrame:
    """Key experiment (§4.6): train on data up to the n-th event, test on the next ``test_days``."""
    cfg = cfg or Config()
    grid = grid or cfg.eval.events_seen_grid
    t0, t1 = d.span()
    ev = d.events[(d.events >= t0) & (d.events < t1)]
    E = P = None
    if representation == "shared":
        E, P = shared_embed_fn(d)
    rows = []
    for n in grid:
        if n is None:                                  # "all": everything but the last block
            tr_end = t1 - test_days * 86400
        elif n == 0:                                   # cold start: just before the first event
            tr_end = ev[0] - 3600
        elif n <= len(ev) - cfg.eval.n_min_events_test:
            tr_end = ev[n - 1] + (cfg.label.post_ictal_h + 1) * 3600
        else:
            continue
        te_end = min(tr_end + test_days * 86400, t1)
        if tr_end <= t0 or ((ev >= tr_end) & (ev < te_end)).sum() == 0:
            continue
        if representation == "features":
            d.fit_normalizer(tr_end)
            X, _ = d.hourly_features()
        else:
            X = E
        f = PatientForecaster(cfg).fit(d, tr_end, X, P)
        for r in score_fold(f, d, tr_end, te_end, X, P, cfg):
            r["events_seen"] = "all" if n is None else n
            rows.append(r)
    return pd.DataFrame(rows)
