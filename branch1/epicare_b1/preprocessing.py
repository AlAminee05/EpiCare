"""Raw E4 sessions -> 60-second windows (guidelines §3, steps 2, 4, 5, 7).

For each patient this produces one ``windows.npz`` with, for every *worn* 60 s window:

    ts       (n,)          int64    window start, Unix seconds UTC, aligned on the minute
    feats    (n, 36)       float32  SeFEF hand-made features (see features.py)
    raw      (n, 6, 60)    float16  compact 1 Hz raw window for the shared network
                                    channels: acc_mean, acc_std, bvp_amp, eda, hr, temp
    cov      (n, 5)        float32  coverage of acc, bvp, eda, hr, temp in the window

plus ``seizures`` (all onsets) and ``events`` (lead-seizure onsets). Windows that are not worn
or miss a required modality are simply absent: a gap is *missing*, never a calm period.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import Config, RAW_CHANNELS
from .features import window_features, FEATURE_NAMES
from .io_msg import Archive, Session, Signal, iter_sessions, load_seizure_times
from .seizures import group_events

COV_MODALITIES = ["acc", "bvp", "eda", "hr", "temp"]
_SIG_KEY = {"acc": "ACC", "bvp": "BVP", "eda": "EDA", "hr": "HR", "temp": "TEMP"}


# ----------------------------------------------------------------------------- filters
def _filter(x: np.ndarray, fs: float, kind: str, cut) -> np.ndarray:
    nyq = fs / 2.0
    if kind == "band":
        lo, hi = cut[0], min(cut[1], 0.9 * nyq)
        if lo >= hi:
            return x - np.mean(x)
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    else:
        c = min(cut, 0.9 * nyq)
        sos = butter(4, c / nyq, btype="low", output="sos")
    if len(x) <= 3 * (2 * sos.shape[0] + 1):
        return x
    return sosfiltfilt(sos, x)


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def clean_session(sess: Session, cfg: Config) -> dict[str, Signal]:
    """Magnitude for ACC, band-pass BVP, low-pass + smooth EDA. Returns 1-D signals."""
    out: dict[str, Signal] = {}
    p = cfg.pre
    s = sess.signals
    if "ACC" in s:
        a = s["ACC"]
        out["acc"] = Signal(a.t0, a.fs, np.linalg.norm(a.values, axis=1))
    if "BVP" in s:
        b = s["BVP"]
        out["bvp"] = Signal(b.t0, b.fs, _filter(b.values, b.fs, "band", p.bvp_band_hz))
    if "EDA" in s:
        e = s["EDA"]
        v = _filter(e.values, e.fs, "low", p.eda_lowpass_hz)
        v = _moving_average(v, int(round(p.eda_smooth_s * e.fs)))
        out["eda"] = Signal(e.t0, e.fs, v)
    for k in ("hr", "temp"):
        if _SIG_KEY[k] in s:
            out[k] = s[_SIG_KEY[k]]
    return out


# ----------------------------------------------------------------------------- windowing
def _window_matrix(sig: Signal, starts: np.ndarray, win_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Gather samples of every window -> (n_win, win_s*fs) with NaN where absent, coverage."""
    fs = sig.fs
    L = int(round(win_s * fs))
    i0 = np.round((starts - sig.t0) * fs).astype(np.int64)
    idx = i0[:, None] + np.arange(L)[None, :]
    ok = (idx >= 0) & (idx < len(sig.values))
    M = np.full(idx.shape, np.nan)
    M[ok] = sig.values[idx[ok]]
    return M, ok.mean(axis=1)


def _per_second(M: np.ndarray, fs: float, win_s: int, how: str) -> np.ndarray:
    """(n, win_s*fs) -> (n, win_s) by per-second mean or std (fs must be ~integer)."""
    f = int(round(fs))
    if f < 1:
        return np.full((M.shape[0], win_s), np.nan)
    R = M[:, : win_s * f].reshape(M.shape[0], win_s, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "std":
            return np.nanstd(R, axis=2) if f > 1 else np.zeros((M.shape[0], win_s))
        return np.nanmean(R, axis=2)


def session_windows(sess: Session, cfg: Config) -> dict[str, np.ndarray] | None:
    sig = clean_session(sess, cfg)
    p = cfg.pre
    W = p.window_s
    if not sig:
        return None
    t_start = min(s.t0 for s in sig.values())
    t_end = max(s.t_end for s in sig.values())
    starts = np.arange(np.ceil(t_start / W) * W, np.floor(t_end / W) * W, W, dtype=np.float64)
    if len(starts) == 0:
        return None
    n = len(starts)
    mats, cov = {}, np.zeros((n, len(COV_MODALITIES)), dtype=np.float32)
    for j, m in enumerate(COV_MODALITIES):
        if m in sig:
            mats[m], cov[:, j] = _window_matrix(sig[m], starts, W)
        else:
            mats[m] = None
    # ---- wear mask: required modalities present + skin warm + EDA non-zero
    need = [COV_MODALITIES.index(m) for m in p.required_modalities]
    valid = (cov[:, need] >= p.min_coverage).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mats["temp"] is not None:
            temp_mean = np.nanmean(mats["temp"], axis=1)
            valid &= ~(temp_mean < p.min_skin_temp_c)  # NaN temp (absent) does not invalidate
        if mats["eda"] is not None:
            valid &= np.nanmedian(mats["eda"], axis=1) >= p.min_eda_us
    if not valid.any():
        return None
    keep = np.flatnonzero(valid)
    sub = {m: (None if M is None else M[keep]) for m, M in mats.items()}
    feats = window_features(sub["acc"], sub["bvp"], sub["eda"], sub["hr"],
                            eda_fs=sig["eda"].fs, eda_prominence=p.eda_peak_prominence_us)
    raw = np.stack([
        _per_second(sub["acc"], sig["acc"].fs, W, "mean"),
        _per_second(sub["acc"], sig["acc"].fs, W, "std"),
        _per_second(sub["bvp"], sig["bvp"].fs, W, "std"),
        _per_second(sub["eda"], sig["eda"].fs, W, "mean"),
        _per_second(sub["hr"], sig["hr"].fs, W, "mean"),
        _per_second(sub["temp"], sig["temp"].fs, W, "mean") if sub["temp"] is not None
        else np.full((len(keep), W), np.nan),
    ], axis=1)
    return {"ts": starts[keep].astype(np.int64), "feats": feats,
            "raw": raw.astype(np.float16), "cov": cov[keep]}


def merge_windows(parts: list[dict]) -> dict[str, np.ndarray]:
    """Concatenate sessions; where two devices overlap keep the window with best coverage."""
    if not parts:
        raise ValueError("no worn windows found")
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    score = d["cov"].sum(axis=1)
    order = np.lexsort((-score, d["ts"]))          # by time, best coverage first
    d = {k: v[order] for k, v in d.items()}
    first = np.r_[True, np.diff(d["ts"]) != 0]
    return {k: v[first] for k, v in d.items()}


def process_archive(archive_path: str | Path, cfg: Config | None = None,
                    out_dir: str | Path | None = None, max_sessions: int | None = None,
                    verbose: bool = True) -> Path:
    """Full preprocessing of one patient archive -> ``<out_dir>/<pid>/windows.npz``."""
    cfg = cfg or Config()
    out_dir = Path(out_dir or cfg.data_processed)
    with Archive(archive_path) as ar:
        pid = ar.patient_id
        seizures = load_seizure_times(ar)
        parts, n_sess, skipped = [], 0, 0
        for sess in iter_sessions(ar):
            n_sess += 1
            if max_sessions and n_sess > max_sessions:
                break
            try:
                w = session_windows(sess, cfg)
            except Exception as e:  # keep going: one broken session must not stop a 6 GB run
                warnings.warn(f"{sess.ref.name}: {e!r}")
                w = None
            if w is None:
                skipped += 1
            else:
                parts.append(w)
            if verbose and n_sess % 10 == 0:
                print(f"\r[{pid}] {n_sess} sessions, {sum(len(p['ts']) for p in parts):,} windows", end="")
    data = merge_windows(parts)
    events, _ = group_events(seizures, cfg.label.event_gap_h)
    dest = out_dir / pid
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "windows.npz", seizures=seizures, events=events, **data)
    meta = {
        "patient": pid, "archive": str(archive_path), "sessions": n_sess, "sessions_skipped": skipped,
        "windows": int(len(data["ts"])), "worn_hours": round(len(data["ts"]) * cfg.pre.window_s / 3600, 1),
        "first_window": int(data["ts"][0]), "last_window": int(data["ts"][-1]),
        "n_seizures": int(len(seizures)), "n_events": int(len(events)),
        "feature_names": FEATURE_NAMES, "raw_channels": RAW_CHANNELS,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"\n[{pid}] done: {meta['windows']:,} worn windows ({meta['worn_hours']} h), "
              f"{meta['n_seizures']} seizures / {meta['n_events']} events -> {dest}")
    return dest / "windows.npz"


def load_processed(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.is_dir():
        path = path / "windows.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
