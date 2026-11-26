"""Branch 1 forecaster for one patient: cycle model + personal layer + fusion (§4).

Wearable representation ``X`` (one row per hour) is either
  * the SeFEF-style hourly features (``PatientData.hourly_features``), the light fallback, or
  * the shared network's hour embeddings (``shared_net.embed``).
``shared_probs`` (the shared network's own heads) is used when the patient does not yet have
enough events for a personal layer (cold start, §4.4).

Components produced for every horizon (1 h, 24 h), all as probabilities:
    base         training event rate only (the "always the average" baseline)
    cycle        personal cycle model only
    wear         personal layer (or calibrated shared head) only       (SeFEF "LR")
    prior_shift  wear with intercept shifted by the cycle log-odds     (SeFEF "LR-vMe")
    epicare      learned fusion of wear + cycle + time since last seizure, fitted on the most
                 recent part of the training period (the final Branch 1 output)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .dataset import PatientData
from .features import FEATURE_NAMES
from .models.cycles import CycleModel
from .models.fusion import (Calibrator, LearnedFusion, PersonalLayer, fusion_inputs,
                            prior_shift_fusion)


def train_index(d: PatientData, t_from: float, t_to: float, horizon_h: int) -> np.ndarray:
    """Hours usable for training: label window inside [t_from, t_to), worn, not post-ictal."""
    ok = (d.hours >= t_from) & (d.hours + horizon_h * 3600 <= t_to) & d.worn & ~d.post_ictal
    return np.flatnonzero(ok)


def _n_events(events, a, b):
    return int(((events >= a) & (events < b)).sum())


@dataclass
class HorizonModel:
    horizon: int
    wear_source: str | None = None          # "personal" | "shared" | None
    personal: PersonalLayer | None = None
    shared_cal: Calibrator | None = None
    wear_prior: float = 0.0
    fusion: LearnedFusion | None = None
    fusion_cycle: CycleModel | None = None   # cycle model fitted on the fusion-fit part only
    base_p: float = 0.0
    mode: str = "base"                       # what "epicare" falls back to


@dataclass
class PatientForecaster:
    cfg: Config = field(default_factory=Config)
    use_personal: bool = True
    feature_names: list[str] | None = None
    cycle: CycleModel | None = None
    models: dict = field(default_factory=dict)
    n_train_events: int = 0
    levels: dict = field(default_factory=dict)
    mean_p1h: float = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, d: PatientData, train_end: float, X: np.ndarray | None = None,
            shared_probs: np.ndarray | None = None) -> "PatientForecaster":
        cfg = self.cfg
        t_start = float(min(d.hours[0], d.seizures[0] if len(d.seizures) else d.hours[0]))
        self.t_start, self.train_end = t_start, float(train_end)
        self.cycle = CycleModel(cfg.cycle, cfg.local_tz).fit(d.events, t_start, train_end)
        self.n_train_events = _n_events(d.events, t_start, train_end)
        hss = d.hours_since_seizure
        for k, h in enumerate(cfg.label.horizons_h):
            m = HorizonModel(h)
            idx = train_index(d, t_start, train_end, h)
            y = d.y[h][idx]
            m.base_p = float(1 - np.exp(-self.cycle.base_rate_h * h))
            # ---- wearable component
            if (self.use_personal and X is not None and len(idx) and
                    self.n_train_events >= cfg.fusion.min_events_personal and 0 < y.sum() < len(y)):
                m.wear_source = "personal"
                m.personal = PersonalLayer(cfg.fusion.personal_C).fit(X[idx], y)
                m.wear_prior = m.personal.train_prior
            elif shared_probs is not None and len(idx) and 0 < y.sum() < len(y):
                m.wear_source = "shared"
                m.shared_cal = Calibrator("platt").fit(shared_probs[idx, k], y)
                m.wear_prior = float(y.mean())
            # ---- learned fusion on the most recent part of the training period
            split = self._split_time(d, t_start, train_end)
            if split is not None:
                cyc_fit = CycleModel(cfg.cycle, cfg.local_tz).fit(d.events, t_start, split)
                fit_idx = train_index(d, t_start, split, h)
                cal_idx = train_index(d, split, train_end, h)
                y_fit, y_cal = d.y[h][fit_idx], d.y[h][cal_idx]
                p_wear_cal = None
                if m.wear_source == "personal" and 0 < y_fit.sum() < len(y_fit):
                    pl = PersonalLayer(cfg.fusion.personal_C).fit(X[fit_idx], y_fit)
                    p_wear_cal = pl.predict_proba(X[cal_idx])
                elif m.wear_source == "shared" and 0 < y_fit.sum() < len(y_fit):
                    p_wear_cal = Calibrator("platt").fit(shared_probs[fit_idx, k], y_fit) \
                        .transform(shared_probs[cal_idx, k])
                if 0 < y_cal.sum() < len(y_cal) and (p_wear_cal is not None or m.wear_source is None):
                    Z = fusion_inputs(p_wear_cal, cyc_fit.predict_proba(d.hours[cal_idx], h), hss[cal_idx])
                    m.fusion = LearnedFusion().fit(Z, y_cal)
                    m.fusion_cycle = cyc_fit
            m.mode = "learned" if m.fusion else ("prior_shift" if m.wear_source else
                                                 ("cycle" if self.cycle.cycles else "base"))
            self.models[h] = m
        # ---- per-patient risk levels and average risk (prior shift for Branch 2)
        tr = train_index(d, t_start, train_end, 1)
        if len(tr):
            comp = self.predict_components(d, tr, X, shared_probs)
            p24 = comp[max(cfg.label.horizons_h)]["epicare"]
            self.levels = {"low_below": float(np.quantile(p24, 0.4)),
                           "high_above": float(np.quantile(p24, 0.8))}
            self.mean_p1h = float(np.mean(comp[1]["epicare"])) if 1 in comp else 0.0
        return self

    def _split_time(self, d: PatientData, t_start: float, train_end: float) -> float | None:
        """Start of the calibration part: last ``calib_frac`` of training, moved earlier if
        needed to hold ``min_calib_events`` events while leaving >= 3 events before it."""
        f = self.cfg.fusion
        split = train_end - f.calib_frac * (train_end - t_start)
        ev = d.events[(d.events >= t_start) & (d.events < train_end)]
        if len(ev) < max(f.min_calib_events + 3, f.min_events_fusion):
            return None
        if _n_events(ev, split, train_end) < f.min_calib_events:
            split = ev[-f.min_calib_events] - 3600
        if _n_events(ev, t_start, split) < 3:
            return None
        return float(split)

    # ------------------------------------------------------------------ predict
    def predict_components(self, d: PatientData, idx: np.ndarray | None = None,
                           X: np.ndarray | None = None, shared_probs: np.ndarray | None = None) -> dict:
        idx = np.arange(len(d.hours)) if idx is None else np.asarray(idx)
        t = d.hours[idx]
        hss = d.hours_since_seizure[idx]
        out = {}
        for k, h in enumerate(self.cfg.label.horizons_h):
            m = self.models[h]
            c = {"base": np.full(len(idx), m.base_p), "cycle": self.cycle.predict_proba(t, h)}
            p_wear = None
            if m.wear_source == "personal":
                p_wear = m.personal.predict_proba(X[idx])
            elif m.wear_source == "shared":
                p_wear = m.shared_cal.transform(shared_probs[idx, k])
            if p_wear is not None:
                c["wear"] = p_wear
                c["prior_shift"] = prior_shift_fusion(p_wear, m.wear_prior, c["cycle"])
            if m.fusion is not None:
                c["epicare"] = m.fusion.predict_proba(fusion_inputs(p_wear, c["cycle"], hss))
            else:
                c["epicare"] = c.get("prior_shift", c["cycle"])
            out[h] = c
        return out

    def predict(self, d: PatientData, idx=None, X=None, shared_probs=None) -> pd.DataFrame:
        idx = np.arange(len(d.hours)) if idx is None else np.asarray(idx)
        comp = self.predict_components(d, idx, X, shared_probs)
        df = pd.DataFrame({"time": pd.to_datetime(d.hours[idx], unit="s", utc=True), "worn": d.worn[idx]})
        for h, c in comp.items():
            for name, v in c.items():
                df[f"{name}_{h}h"] = v
            df[f"y_{h}h"] = d.y[h][idx]
        return df

    # ------------------------------------------------------------------ app output (§4.5)
    def hourly_output(self, d: PatientData, i: int, X=None, shared_probs=None) -> dict:
        H = max(self.cfg.label.horizons_h)
        comp = self.predict_components(d, np.array([i]), X, shared_probs)
        p24 = float(comp[H]["epicare"][0])
        p1 = float(comp[1]["epicare"][0]) if 1 in comp else p24 / H
        base24 = self.models[H].base_p
        lv = self.levels or {"low_below": base24 * 0.7, "high_above": base24 * 1.5}
        level = "high" if p24 >= lv["high_above"] else ("low" if p24 < lv["low_below"] else "medium")
        r_bar = self.mean_p1h or self.models[1].base_p
        shift = float(np.clip(np.log(max(p1, 1e-9) / max(r_bar, 1e-9)),
                              -self.cfg.fusion.prior_shift_cap, self.cfg.fusion.prior_shift_cap))
        t = float(d.hours[i])
        drivers = self.cycle.drivers(t)
        if d.hours_since_seizure[i] < 24:
            drivers.append("seizure in the last 24 h")
        m = self.models[H]
        if m.wear_source == "personal" and X is not None and self.feature_names:
            for name, _ in m.personal.top_features(self.feature_names, X[i], k=3):
                if not name.startswith("coverage"):   # wear time is not a reason to show a patient
                    drivers.append(readable_feature(name))
        conf = "low" if self.n_train_events < 5 else ("medium" if self.n_train_events < 20 else "high")
        # 24 h profile: shape from the cycle model, total matched to p24
        offs = t + np.arange(H) * 3600.0
        lam = self.cycle.rate(offs)
        lam = lam * (-np.log(1 - min(p24, 1 - 1e-9)) / max(lam.sum(), 1e-12))
        if p24 >= 0.25:
            msg = f"About {max(1, round(p24 * 10))} days in 10 like today have a seizure"
        else:
            msg = f"About 1 day in {int(round(1 / max(p24, 1e-9)))} like today has a seizure"
        return {
            "time": pd.to_datetime(t, unit="s", utc=True).tz_convert(self.cfg.local_tz).strftime("%Y-%m-%dT%H:%M"),
            "horizon": f"{H}h", "probability": round(p24, 3), "probability_next_hour": round(p1, 4),
            "baseline_probability": round(base24, 3), "level": level,
            "prior_shift_for_detector": round(shift, 2), "drivers": drivers, "confidence": conf,
            "message": msg,
            "profile_24h": [round(float(1 - np.exp(-x)), 4) for x in lam],
        }

    def doctor_summary(self) -> dict:
        return {"n_training_events": self.n_train_events, **self.cycle.summary(),
                "fusion": {h: (m.mode, m.fusion.weights if m.fusion else None) for h, m in self.models.items()},
                "levels": self.levels}


def readable_feature(name: str) -> str:
    """'f13_last6h' -> 'bvp std above usual (last 6 h)'."""
    if name.startswith("f") and "_last" in name:
        j, b = name[1:].split("_last")
        try:
            return f"{FEATURE_NAMES[int(j)].replace('_', ' ')} unusual (last {b.rstrip('h')} h)"
        except (ValueError, IndexError):
            pass
    return name


def to_json(out: dict) -> str:
    return json.dumps(out, indent=1)
