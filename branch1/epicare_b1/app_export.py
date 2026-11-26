"""Export Branch 1 results for the EpiCare mobile app (``app/``).

One JSON file holds everything the app shows. Its top level is shared by all branches, so
Branch 2 can later add its own block next to ``branches.branch1`` and the app merges them:

    {
      "schema": "epicare.app/1",
      "generated": "...", "dataset": {...},
      "branches": {"branch1": {...}, "branch2": null}
    }

Per patient, Branch 1 exports:
    summary      recording length, wear, seizures, events
    current      the hourly output of guidelines §4.5 (final model, latest worn hour)
    timeline     out-of-sample forecasts of the CV test blocks (last ``timeline_days`` days)
                 with the seizures that really happened, so the app can show both
    prior_shift  the value sent to Branch 2 over the last week (final model)
    cycles       the personal cycles found on all data
    metrics      mean over CV folds, per horizon and model (AUC, BSS, false alarms/day)
    reliability  calibration of the 1 h EpiCare forecast on all CV test blocks
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, SEFEF_BASELINE_AUC
from .dataset import PatientData
from .evaluation import reliability_table
from .models.cycles import CycleModel
from .pipeline import fit_final
from .seizures import summarize_seizures

SCHEMA = "epicare.app/1"


def _num(x, nd=4):
    """JSON-safe rounded float (NaN -> None)."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def _iso(t: float, tz: str) -> str:
    return pd.to_datetime(float(t), unit="s", utc=True).tz_convert(tz).strftime("%Y-%m-%dT%H:%M")


def _metrics(res: pd.DataFrame, pid: str | None = None) -> dict:
    r = res if pid is None else res[res.patient == pid]
    out = {}
    for (h, model), g in r.groupby(["horizon_h", "model"]):
        out.setdefault(f"{int(h)}h", {})[model] = {
            "auc": _num(g["AUC_TiW"].mean(), 3), "auc_sd": _num(g["AUC_TiW"].std(), 3),
            "bss": _num(g["BSS"].mean(), 3), "fa_per_day": _num(g["FA_per_day"].mean(), 3),
            "folds": int(g["fold"].nunique()) if "fold" in g else int(len(g)),
        }
    return out


def _unix(times: pd.Series) -> np.ndarray:
    """UTC datetimes -> Unix seconds, whatever the datetime resolution (pandas 2 or 3)."""
    return ((times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def _timeline(pred: pd.DataFrame, d: PatientData, levels: dict, days: int, tz: str) -> dict:
    pred = pred.tail(days * 24)
    t = _unix(pred["time"])
    lo, hi = (t[0], t[-1] + 3600) if len(t) else (0, 0)
    seiz = d.seizures[(d.seizures >= lo) & (d.seizures < hi)]
    return {
        "start": _iso(lo, tz) if len(t) else None,
        "step_h": 1,
        "p24": [_num(x) for x in pred["epicare_24h"]],
        "p1": [_num(x, 5) for x in pred["epicare_1h"]],
        "cycle_p1": [_num(x, 5) for x in pred["cycle_1h"]],
        "worn": [bool(x) for x in pred["worn"]],
        "seizures": [_iso(s, tz) for s in seiz],
        "levels": {k: _num(v) for k, v in (levels or {}).items()},
        "out_of_sample": True,
    }


def _prior_shift_week(final, d: PatientData, X, tz: str) -> dict:
    idx = np.arange(max(0, len(d.hours) - 24 * 7), len(d.hours))
    comp = final.predict_components(d, idx, X)
    p1 = comp[1]["epicare"]
    r_bar = final.mean_p1h or final.models[1].base_p
    cap = final.cfg.fusion.prior_shift_cap
    shift = np.clip(np.log(np.maximum(p1, 1e-9) / max(r_bar, 1e-9)), -cap, cap)
    return {"start": _iso(d.hours[idx[0]], tz), "step_h": 1, "values": [_num(x, 3) for x in shift],
            "cap": cap}


def patient_block(d: PatientData, res: pd.DataFrame, fitted: list, cfg: Config,
                  timeline_days: int = 14) -> dict:
    tz = cfg.local_tz
    s = d.summary()
    sz = summarize_seizures(d.seizures, cfg.label.event_gap_h, tz)
    block = {
        "id": d.pid,
        "summary": {**s, "first": _iso(d.hours[0], tz), "last": _iso(d.hours[-1], tz),
                    "events_per_day": sz["events_per_day"]},
        "event_hour_hist": sz["event_hour_local_hist"],
    }
    cm = CycleModel(cfg.cycle, tz).fit(d.events, d.hours[0], d.hours[-1])
    block["cycles"] = cm.summary()
    # final model on all data: "current" hourly output + prior shift for Branch 2
    final = fit_final(d, cfg)
    X, _ = d.hourly_features()
    worn = np.flatnonzero(d.worn)
    i = int(worn[-1]) if len(worn) else len(d.hours) - 1
    block["current"] = final.hourly_output(d, i, X)
    block["prior_shift"] = _prior_shift_week(final, d, X, tz)
    block["model"] = {"n_training_events": final.n_train_events,
                      "modes": {f"{h}h": m.mode for h, m in final.models.items()},
                      "wear_source": {f"{h}h": m.wear_source for h, m in final.models.items()}}
    # cross-validation results
    block["metrics"] = _metrics(res, d.pid)
    if fitted:
        # every fold's test block, each predicted by the model fitted on the past only
        pred = pd.concat([f[2] for f in fitted], ignore_index=True).sort_values("time")
        pred = pred.drop_duplicates("time", keep="last").reset_index(drop=True)
        block["timeline"] = _timeline(pred, d, fitted[-1][1].levels, timeline_days, tz)
        post = d.post_ictal[np.searchsorted(d.hours, _unix(pred["time"]))]
        sel = pred["worn"].to_numpy() & ~post
        rel = reliability_table(pred.loc[sel, "epicare_1h"], pred.loc[sel, "y_1h"], cfg.eval.n_reliability_bins)
        block["reliability_1h"] = [{"p": _num(r.p_mean, 5), "obs": _num(r.observed, 5), "n": int(r.n)}
                                   for r in rel.itertuples()]
    return block


def export_app_data(P: dict[str, PatientData], res: pd.DataFrame, fitted: dict, cfg: Config | None = None,
                    out_path: str | Path = "results/app_data.json", synthetic: bool = False,
                    representation: str = "features", events_seen: pd.DataFrame | None = None,
                    merge_into_existing: bool = True) -> Path:
    """Write the app JSON. If the file exists, other branches' blocks are kept."""
    cfg = cfg or Config()
    out_path = Path(out_path)
    b1 = {
        "title": "Seizure risk forecasting",
        "representation": representation,
        "horizons_h": list(cfg.label.horizons_h),
        "local_tz": cfg.local_tz,
        "patients": [patient_block(d, res, fitted.get(pid, []), cfg) for pid, d in P.items()],
        "metrics": _metrics(res),
        "sefef_baseline_auc": SEFEF_BASELINE_AUC,
    }
    if events_seen is not None and not events_seen.empty:
        g = events_seen.groupby(["horizon_h", "model", "events_seen"], sort=False)[["AUC_TiW", "BSS"]].mean()
        b1["events_seen"] = [{"horizon_h": int(h), "model": m, "events": str(n),
                              "auc": _num(r.AUC_TiW, 3), "bss": _num(r.BSS, 3)}
                             for (h, m, n), r in g.iterrows()]
    doc = {}
    if merge_into_existing and out_path.exists():
        try:
            doc = json.loads(out_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            doc = {}
    doc.update({
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {"name": "My Seizure Gauge (Mayo Clinic, Zenodo 17380899)" if not synthetic
                    else "Synthetic archives (code check only)",
                    "synthetic": bool(synthetic), "patients": list(P)},
    })
    doc.setdefault("branches", {}).setdefault("branch2", None)
    doc["branches"]["branch1"] = b1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out_path
