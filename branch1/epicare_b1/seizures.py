"""Seizure times -> events, plus the time helpers used everywhere."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_events(onsets: np.ndarray, gap_h: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Group seizures into events (guidelines §3 step 3).

    A seizure starts a new event only if the previous seizure was at least ``gap_h`` hours
    earlier. Returns ``(event_onsets, event_id_per_seizure)``. The event onset is the time of
    its first ("lead") seizure, which is the SeFEF definition of a lead seizure as well.
    """
    t = np.sort(np.asarray(onsets, dtype=np.float64))
    if len(t) == 0:
        return t, np.zeros(0, dtype=int)
    new = np.r_[True, np.diff(t) >= gap_h * 3600.0]
    ids = np.cumsum(new) - 1
    return t[new], ids


def summarize_seizures(onsets: np.ndarray, gap_h: float = 4.0, tz: str = "America/Chicago") -> dict:
    t = np.sort(np.asarray(onsets, dtype=np.float64))
    ev, _ = group_events(t, gap_h)
    span_d = (t[-1] - t[0]) / 86400 if len(t) > 1 else 0.0
    local = pd.to_datetime(ev, unit="s", utc=True).tz_convert(tz)
    return {
        "n_seizures": int(len(t)),
        "n_events": int(len(ev)),
        "first": str(pd.to_datetime(t[0], unit="s", utc=True)) if len(t) else None,
        "last": str(pd.to_datetime(t[-1], unit="s", utc=True)) if len(t) else None,
        "span_days": round(span_d, 1),
        "events_per_day": round(len(ev) / span_d, 3) if span_d else None,
        "event_hour_local_hist": np.bincount(local.hour, minlength=24).tolist(),
    }


def hours_since_last(times: np.ndarray, seizure_onsets: np.ndarray, cap_h: float = 24 * 30) -> np.ndarray:
    """Hours since the most recent seizure strictly before each time (capped)."""
    s = np.sort(np.asarray(seizure_onsets, dtype=np.float64))
    idx = np.searchsorted(s, times, side="left") - 1
    out = np.full(len(times), cap_h, dtype=np.float64)
    ok = idx >= 0
    out[ok] = np.minimum((times[ok] - s[idx[ok]]) / 3600.0, cap_h)
    return out


def to_local(ts: np.ndarray, tz: str) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="s", utc=True).tz_convert(tz)
