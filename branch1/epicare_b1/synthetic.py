"""Synthetic patient archives in the *same layout* as the My Seizure Gauge release.

Purpose: run and test the full Branch 1 pipeline before the 3-7 GB real archives are downloaded.
The fake archive is ``Mayo_<id>.zip`` containing ``Mayo_<id>/<id>.txt`` (seizure onsets, one
Unix timestamp per line) and nested Empatica-E4-style session zips ``<start>_<serial>.zip``
with ACC/BVP/EDA/HR/TEMP/IBI/tags CSVs and info.txt.

Built-in ground truth, so you can check the models find it:
  * seizure rate with a daily cycle (peak at ``peak_hour`` local time) and a multiday cycle
    (``multiday_period_d``), plus clusters (follow-up seizures within a few hours);
  * a pre-ictal physiological change during the ``preictal_h`` hours before each event:
    heart rate up, skin conductance up, less movement.
To keep files small the default rates are lower than a real E4 (the loader reads the rate from
each file, so this changes nothing in the code path).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .seizures import group_events


@dataclass
class SynthConfig:
    patient_id: str = "9001"
    start: str = "2020-01-06"        # local date of the first day
    n_days: int = 75
    tz: str = "America/Chicago"
    events_per_day: float = 0.7
    peak_hour: float = 15.0          # daily cycle peak (local); daytime so the wearable sees it
    circadian_kappa: float = 1.2
    multiday_period_d: float = 7.0
    multiday_kappa: float = 1.0
    cluster_prob: float = 0.3
    preictal_h: float = 2.0
    hr_effect: float = 3.0           # bpm (small next to the noise)
    eda_effect: float = 0.3          # microsiemens
    session_h: tuple = (7.0, 11.0)   # worn hours per day (random in range)
    p_day_missing: float = 0.1
    acc_fs: int = 4
    bvp_fs: int = 8
    eda_fs: int = 4
    temp_fs: int = 4
    seed: int = 0


def _seizure_times(cfg: SynthConfig, rng) -> np.ndarray:
    t0 = pd.Timestamp(cfg.start, tz=cfg.tz).value // 10 ** 9
    hours = t0 + np.arange(cfg.n_days * 24) * 3600
    loc_h = pd.to_datetime(hours, unit="s", utc=True).tz_convert(cfg.tz).hour.to_numpy()
    th_c = 2 * np.pi * (loc_h - cfg.peak_hour) / 24
    th_m = 2 * np.pi * ((hours - t0) / 86400 % cfg.multiday_period_d) / cfg.multiday_period_d
    w = np.exp(cfg.circadian_kappa * np.cos(th_c)) * np.exp(cfg.multiday_kappa * np.cos(th_m - 1.0))
    lam = cfg.events_per_day / 24 * w / w.mean()
    n = rng.poisson(lam)
    ev = np.concatenate([h + rng.uniform(0, 3600, k) for h, k in zip(hours, n) if k])
    ev, _ = group_events(ev, 4.0)  # keep lead seizures 4 h apart, then add clusters
    sz = [ev]
    for e in ev:
        if rng.random() < cfg.cluster_prob:
            sz.append(e + np.sort(rng.uniform(600, 3 * 3600, rng.integers(1, 3))))
    return np.sort(np.concatenate(sz)).round()


def _session(cfg: SynthConfig, start: int, dur_s: int, events: np.ndarray, rng) -> dict[str, np.ndarray]:
    """Per-second physiology, then upsampled to each sensor rate."""
    t = start + np.arange(dur_s)
    loc_h = pd.to_datetime(t[::60], unit="s", utc=True).tz_convert(cfg.tz).hour.to_numpy().repeat(60)[:dur_s]
    # activity: minute-level Markov chain rest(0)/light(1)/active(2)
    n_min = dur_s // 60 + 1
    state = np.zeros(n_min, int)
    P = np.array([[0.9, 0.08, 0.02], [0.15, 0.75, 0.10], [0.1, 0.3, 0.6]])
    for i in range(1, n_min):
        state[i] = rng.choice(3, p=P[state[i - 1]])
    act = np.array([0.02, 0.15, 0.5])[state].repeat(60)[:dur_s]
    # pre-ictal ramp: 0 -> 1 over the preictal_h hours before each upcoming event
    nxt = np.searchsorted(events, t, side="left")
    dt = np.where(nxt < len(events), events[np.minimum(nxt, len(events) - 1)] - t, np.inf)
    pre = np.clip(1 - dt / (cfg.preictal_h * 3600), 0, 1)
    act = act * (1 - 0.2 * pre)
    hr = 68 + 25 * act + 4 * np.sin(2 * np.pi * (loc_h - 16) / 24) + cfg.hr_effect * pre \
        + np.cumsum(rng.normal(0, 0.05, dur_s)) * 0.1 + rng.normal(0, 1.5, dur_s)
    eda = 1.5 + 0.6 * np.sin(2 * np.pi * t / 5400) ** 2 + cfg.eda_effect * pre \
        + np.abs(np.cumsum(rng.normal(0, 0.003, dur_s)))
    # skin conductance responses
    for s in rng.choice(dur_s, size=max(1, dur_s // 400), replace=False):
        k = np.arange(min(20, dur_s - s))
        eda[s: s + len(k)] += 0.15 * (1 - np.exp(-k / 2)) * np.exp(-k / 8)
    temp = 33 + 0.5 * np.sin(2 * np.pi * (loc_h - 18) / 24) + rng.normal(0, 0.05, dur_s)
    up = lambda x, fs: np.repeat(x, fs)
    # ACC (1/64 g, integers): gravity on z + movement noise
    fa = cfg.acc_fs
    mov = up(act, fa) * 64
    acc = np.stack([rng.normal(0, 1, dur_s * fa) * mov,
                    rng.normal(0, 1, dur_s * fa) * mov,
                    64 + rng.normal(0, 1, dur_s * fa) * mov], 1).round().astype(int)
    fb = cfg.bvp_fs
    tb = np.arange(dur_s * fb) / fb
    phase = 2 * np.pi * np.cumsum(up(hr / 60.0, fb)) / fb
    bvp = (40 * np.sin(phase) * (1 + up(act, fb)) + rng.normal(0, 5, len(tb))).round(2)
    return {"ACC": acc, "BVP": bvp, "EDA": up(eda, cfg.eda_fs).round(4),
            "HR": hr.round(2), "TEMP": up(temp, cfg.temp_fs).round(2)}


def _csv(start: float, fs: float, values: np.ndarray) -> bytes:
    buf = io.StringIO()
    ncol = values.shape[1] if values.ndim == 2 else 1
    buf.write(",".join([f"{start:.6f}"] * ncol) + "\n")
    buf.write(",".join([f"{fs:.6f}"] * ncol) + "\n")
    if values.ndim == 2:   # fast path: str() of Python ints/floats beats np.savetxt ~5x
        buf.write("\n".join(",".join(map(str, r)) for r in values.tolist()))
    else:
        buf.write("\n".join(map(str, values.tolist())))
    return (buf.getvalue() + "\n").encode()


def make_archive(out_dir: str | Path, cfg: SynthConfig | None = None, verbose: bool = True) -> Path:
    """Write ``<out_dir>/Mayo_<id>.zip`` and return its path."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seizures = _seizure_times(cfg, rng)
    events, _ = group_events(seizures, 4.0)
    pid = cfg.patient_id
    path = out_dir / f"Mayo_{pid}.zip"
    serials = ["A0F001", "A0F002"]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as outer:
        # seizure file deliberately unsorted, like the real ones can be
        shuffled = seizures.copy()
        rng.shuffle(shuffled)
        outer.writestr(f"Mayo_{pid}/{pid}.txt", "\n".join(str(int(s)) for s in shuffled) + "\n")
        for day in range(cfg.n_days):
            if rng.random() < cfg.p_day_missing:
                continue
            d0 = pd.Timestamp(cfg.start, tz=cfg.tz) + pd.Timedelta(days=day)
            start = int((d0 + pd.Timedelta(hours=float(rng.uniform(7, 10)))).value // 10 ** 9)
            dur = int(rng.uniform(*cfg.session_h) * 3600)
            sig = _session(cfg, start, dur, events, rng)
            serial = serials[day % 2]
            inner_buf = io.BytesIO()
            with zipfile.ZipFile(inner_buf, "w", zipfile.ZIP_DEFLATED) as inner:
                inner.writestr("ACC.csv", _csv(start, cfg.acc_fs, sig["ACC"]))
                inner.writestr("BVP.csv", _csv(start, cfg.bvp_fs, sig["BVP"]))
                inner.writestr("EDA.csv", _csv(start, cfg.eda_fs, sig["EDA"]))
                inner.writestr("HR.csv", _csv(start + 10, 1, sig["HR"][10:]))
                inner.writestr("TEMP.csv", _csv(start, cfg.temp_fs, sig["TEMP"]))
                inner.writestr("IBI.csv", f"{start:.6f}, IBI\n")
                inner.writestr("tags.csv", "")
                inner.writestr("info.txt", "Synthetic E4-format session (EpiCare test data)\n")
            outer.writestr(f"Mayo_{pid}/{start}_{serial}.zip", inner_buf.getvalue())
            if verbose:
                print(f"\r[{pid}] day {day + 1}/{cfg.n_days}", end="")
        # a non-E4 session, to check the loader skips unknown devices gracefully
        fb = io.BytesIO()
        with zipfile.ZipFile(fb, "w") as z:
            z.writestr("steps.json", "[]")
        outer.writestr(f"Mayo_{pid}/{int(pd.Timestamp(cfg.start, tz=cfg.tz).value // 10**9)}_000000.zip", fb.getvalue())
    if verbose:
        print(f"\n[{pid}] wrote {path} ({path.stat().st_size / 1e6:.1f} MB): "
              f"{len(seizures)} seizures, {len(events)} events")
    return path
