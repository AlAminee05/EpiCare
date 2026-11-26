"""Per-patient hourly view of the windows: labels, wear mask, normalisation, features.

Branch 1 issues one forecast at the start of every clock hour ``t``. Everything the forecast
uses comes from *before* ``t``; the targets look *after* ``t``:

    y_1h[t]  = 1 if an event (lead seizure) starts in [t, t + 1 h)
    y_24h[t] = 1 if an event starts in [t, t + 24 h)
    worn[t]  = at least ``min_hour_coverage`` of the windows in [t - 1 h, t) are worn
    post_ictal[t] = a seizure happened in (t - post_ictal_h, t]   (removed from training)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .seizures import to_local, hours_since_last


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class HourOfDayNormalizer:
    """Robust z-score against the patient's own baseline for each local hour of day (§3 step 7)."""
    med: np.ndarray | None = None   # (24, d)
    iqr: np.ndarray | None = None
    clip: float = 5.0

    def fit(self, X: np.ndarray, hod: np.ndarray) -> "HourOfDayNormalizer":
        X = signed_log(X.astype(np.float64))
        d = X.shape[1]
        g_med = np.nanmedian(X, axis=0)
        g_iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
        self.med = np.tile(g_med, (24, 1))
        self.iqr = np.tile(g_iqr, (24, 1))
        for h in range(24):
            sel = X[hod == h]
            if len(sel) >= 30:  # enough windows at this hour, otherwise keep the global value
                self.med[h] = np.nanmedian(sel, axis=0)
                self.iqr[h] = np.nanpercentile(sel, 75, axis=0) - np.nanpercentile(sel, 25, axis=0)
        self.iqr = np.where(np.isfinite(self.iqr) & (self.iqr > 1e-6), self.iqr, 1.0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        assert self.med.shape == (24, d)
        return self

    def transform(self, X: np.ndarray, hod: np.ndarray) -> np.ndarray:
        Z = (signed_log(X.astype(np.float64)) - self.med[hod]) / self.iqr[hod]
        return np.nan_to_num(np.clip(Z, -self.clip, self.clip)).astype(np.float32)


class PatientData:
    """All of one patient's data on a common hourly grid."""

    def __init__(self, pid: str, windows: dict, cfg: Config | None = None):
        self.pid = str(pid)
        self.cfg = cfg = cfg or Config()
        W = cfg.pre.window_s
        self.ts = windows["ts"].astype(np.int64)
        self.feats = windows["feats"].astype(np.float32)
        self.raw = windows["raw"].astype(np.float16)           # (n, C, 60), kept compact
        self.seizures = np.sort(windows["seizures"].astype(np.float64))
        self.events = np.sort(windows["events"].astype(np.float64))
        self.hod_win = to_local(self.ts, cfg.local_tz).hour.to_numpy()

        # ---- hourly grid covering the wearable recording
        h0 = (self.ts[0] // 3600) * 3600 + 3600     # first hour with a full past hour
        h1 = (self.ts[-1] // 3600) * 3600 + 3600
        self.hours = np.arange(h0, h1 + 1, 3600, dtype=np.int64)
        loc = to_local(self.hours, cfg.local_tz)
        self.hod = loc.hour.to_numpy()
        self.local_date = loc.date

        # ---- dense minute index: minute k -> row in the window arrays, -1 = not worn
        self.m0 = int(self.ts[0] // W)
        n_min = int(self.ts[-1] // W) - self.m0 + 1
        self.minute_index = np.full(n_min, -1, dtype=np.int64)
        self.minute_index[(self.ts // W - self.m0).astype(np.int64)] = np.arange(len(self.ts))

        per_hour = 3600 // W
        self.coverage = self.window_counts(self.hours - 3600, self.hours) / per_hour
        self.worn = self.coverage >= cfg.label.min_hour_coverage
        self.y = {h: self._label(h) for h in cfg.label.horizons_h}
        self.post_ictal = self._post_ictal(cfg.label.post_ictal_h)
        self.hours_since_seizure = hours_since_last(self.hours.astype(float), self.seizures)
        self.feats_n = self.raw_n = None
        self.normalizer_end = None

    # ------------------------------------------------------------------ labels / masks
    def _label(self, horizon_h: int) -> np.ndarray:
        lo = np.searchsorted(self.events, self.hours, side="left")
        hi = np.searchsorted(self.events, self.hours + horizon_h * 3600, side="left")
        return (hi > lo).astype(np.int8)

    def _post_ictal(self, post_h: float) -> np.ndarray:
        lo = np.searchsorted(self.seizures, self.hours - post_h * 3600, side="right")
        hi = np.searchsorted(self.seizures, self.hours, side="right")
        return hi > lo

    def window_counts(self, t_from: np.ndarray, t_to: np.ndarray) -> np.ndarray:
        lo = np.searchsorted(self.ts, t_from, side="left")
        hi = np.searchsorted(self.ts, t_to, side="left")
        return (hi - lo).astype(np.float64)

    # ------------------------------------------------------------------ normalisation
    def fit_normalizer(self, train_end: float | None = None) -> "PatientData":
        """Fit the hour-of-day baseline on windows before ``train_end`` only (no leakage)."""
        sel = self.ts < (train_end if train_end is not None else np.inf)
        if sel.sum() < 100:  # too little history: use the first day of recording only
            sel = np.arange(len(self.ts)) < min(len(self.ts), 1440)
        self.f_norm = HourOfDayNormalizer().fit(self.feats[sel], self.hod_win[sel])
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rmean = np.nanmean(self.raw.astype(np.float32), axis=2)  # channel baseline = window mean
        # the network sees signed-log raw values; the baseline is therefore fitted on log(mean)
        self.r_norm = HourOfDayNormalizer().fit(rmean[sel], self.hod_win[sel])
        self.feats_n = self.f_norm.transform(self.feats, self.hod_win)
        # raw: same per-channel shift/scale for every second of the window, in chunks (memory)
        self.raw_n = np.empty(self.raw.shape, dtype=np.float16)
        for a in range(0, len(self.raw), 50_000):
            b = a + 50_000
            R = signed_log(self.raw[a:b].astype(np.float32))
            med = self.r_norm.med[self.hod_win[a:b]][:, :, None]
            iqr = self.r_norm.iqr[self.hod_win[a:b]][:, :, None]
            self.raw_n[a:b] = np.nan_to_num(np.clip((R - med) / iqr, -5, 5))
        self.normalizer_end = train_end
        return self

    # ------------------------------------------------------------------ hourly features
    def hourly_features(self, blocks_h: tuple = (1, 6, 24)) -> tuple[np.ndarray, list[str]]:
        """Mean normalised window features over the last 1 h, 6 h and 24 h, plus coverage.

        This is the SeFEF-style "lighter fallback" representation of the wearable signals.
        Missing hours get 0 (= the patient's usual value) and a low coverage value.
        """
        if self.feats_n is None:
            raise RuntimeError("call fit_normalizer(train_end) first")
        csum = np.vstack([np.zeros((1, self.feats_n.shape[1])), np.cumsum(self.feats_n, axis=0)])
        cols, names = [], []
        for b in blocks_h:
            lo = np.searchsorted(self.ts, self.hours - b * 3600, side="left")
            hi = np.searchsorted(self.ts, self.hours, side="left")
            n = (hi - lo)[:, None]
            m = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), 0.0)
            cols.append(m)
            names += [f"f{j}_last{b}h" for j in range(self.feats_n.shape[1])]
        cols.append(self.coverage[:, None])
        names.append("coverage_1h")
        return np.hstack(cols).astype(np.float32), names

    # ------------------------------------------------------------------ helpers
    def utc_offset(self, t: np.ndarray) -> np.ndarray:
        """Seconds to add to UTC to get local wall-clock time (handles DST)."""
        loc = to_local(np.asarray(t), self.cfg.local_tz)
        return (loc.tz_localize(None).asi8 // 10 ** 9) - np.asarray(t, dtype=np.int64)

    def span(self) -> tuple[float, float]:
        return float(self.hours[0]), float(self.hours[-1])

    def summary(self) -> dict:
        return {
            "patient": self.pid,
            "days": round((self.hours[-1] - self.hours[0]) / 86400, 1),
            "worn_fraction": round(float(self.worn.mean()), 3),
            "n_seizures": int(len(self.seizures)),
            "n_events": int(len(self.events)),
            "events_in_recording": int(((self.events >= self.hours[0]) & (self.events <= self.hours[-1])).sum()),
            "pos_rate_1h": round(float(self.y[1].mean()), 4) if 1 in self.y else None,
            "pos_rate_24h": round(float(self.y[24].mean()), 4) if 24 in self.y else None,
        }
