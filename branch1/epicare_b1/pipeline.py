"""High-level functions: what the notebook and the command line call."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, SEFEF_BASELINE_AUC
from .dataset import PatientData
from .evaluation import evaluate_patient, summarize, tscv_splits
from .forecaster import PatientForecaster
from .preprocessing import load_processed


def load_patients(processed_dir: str | Path, ids: list[str] | None = None,
                  cfg: Config | None = None) -> dict[str, PatientData]:
    cfg = cfg or Config()
    processed_dir = Path(processed_dir)
    ids = ids or sorted(p.name for p in processed_dir.iterdir() if (p / "windows.npz").exists())
    return {pid: PatientData(pid, load_processed(processed_dir / pid), cfg) for pid in ids}


def first_fold_end(P: dict[str, PatientData], cfg: Config) -> dict[str, float]:
    """End of each patient's first training block: the shared network may only see data before
    it when the same patient's later folds are used for testing."""
    out = {}
    for pid, d in P.items():
        f = tscv_splits(d, cfg)
        out[pid] = f[0][0] if f else float(d.hours[-1]) + 1
    return out


def train_shared_network(P: dict[str, PatientData], cfg: Config | None = None,
                         train_end: dict[str, float] | None = None, verbose: bool = True):
    """Self-supervised pretraining + light supervised step on all given patients."""
    from .models import shared_net as sn
    import torch

    cfg = cfg or Config()
    torch.manual_seed(cfg.net.seed)
    train_end = train_end or first_fold_end(P, cfg)
    for pid, d in P.items():
        d.fit_normalizer(train_end[pid])
    batcher = sn.HourBatcher(P, cfg.net, cfg.pre.window_s)
    model = sn.SharedNet(cfg.net).to(cfg.net.device)
    hist = {"pretrain": sn.pretrain(model, batcher, train_end, verbose=verbose),
            "finetune": sn.finetune(model, batcher, train_end, verbose=verbose)}
    return model, batcher, hist


def run_branch1(P: dict[str, PatientData], cfg: Config | None = None, representation: str = "features",
                model=None, batcher=None, verbose: bool = True):
    """Time-series CV for every patient. Returns (per-fold results, summary, fitted forecasters)."""
    cfg = cfg or Config()
    embed_fn = None
    if representation == "shared":
        from .models import shared_net as sn
        embed_fn = lambda d: sn.embed(model, batcher, d.pid)
    rows, fitted = [], {}
    for pid, d in P.items():
        r, f = evaluate_patient(d, cfg, representation, embed_fn, verbose)
        rows.append(r)
        fitted[pid] = f
    res = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return res, summarize(res), fitted


def leave_one_patient_out(P: dict[str, PatientData], cfg: Config | None = None, verbose: bool = True):
    """New-patient test (§4.6): the shared network never sees the test patient."""
    cfg = cfg or Config()
    from .models import shared_net as sn
    rows = []
    for test_pid in P:
        others = {k: v for k, v in P.items() if k != test_pid}
        ends = {k: float(v.hours[-1]) + 1 for k, v in others.items()}  # all data of the others
        model, _, _ = train_shared_network(others, cfg, ends, verbose=False)
        d = P[test_pid]
        d.fit_normalizer(first_fold_end({test_pid: d}, cfg)[test_pid])
        b = sn.HourBatcher({test_pid: d}, cfg.net, cfg.pre.window_s)
        r, _ = evaluate_patient(d, cfg, "shared", lambda dd: sn.embed(model, b, dd.pid), verbose)
        r["setting"] = "LOPO"
        rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    return res, summarize(res)


def compare_with_sefef(summary: pd.DataFrame) -> pd.DataFrame:
    """Our mean AUC next to the published SeFEF numbers (same six patients when applicable)."""
    rows = []
    for (h, model), r in summary.iterrows():
        key = "hourly" if h == 1 else "daily"
        rows.append({"horizon": key, "model": model, "our_AUC": r.get("AUC_TiW_mean"),
                     "our_BSS": r.get("BSS_mean"),
                     "SeFEF_vMe_AUC": SEFEF_BASELINE_AUC[key]["vMe"],
                     "SeFEF_LR_AUC": SEFEF_BASELINE_AUC[key]["LR"]})
    return pd.DataFrame(rows)


def fit_final(d: PatientData, cfg: Config | None = None, X=None, shared_probs=None,
              train_end: float | None = None) -> PatientForecaster:
    """Fit on all available data (deployment / demo replay)."""
    cfg = cfg or Config()
    train_end = train_end or float(d.hours[-1]) + 1
    if X is None and shared_probs is None:
        d.fit_normalizer(train_end)
        X, names = d.hourly_features()
    else:
        names = [f"emb{j}" for j in range(X.shape[1])] if X is not None else None
    return PatientForecaster(cfg, feature_names=names).fit(d, train_end, X, shared_probs)


def save_forecaster(f: PatientForecaster, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(f, fh)


def load_forecaster(path: str | Path) -> PatientForecaster:
    with open(path, "rb") as fh:
        return pickle.load(fh)
