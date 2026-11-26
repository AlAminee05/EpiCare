"""Figures for the notebook and the report."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import PatientData
from .evaluation import auc_sen_tiw, reliability_table
from .seizures import to_local


def plot_wear_and_seizures(d: PatientData, ax=None):
    """Daily wear time (bars) and seizure / event times (marks): shows the gaps in the data."""
    ax = ax or plt.subplots(figsize=(12, 3))[1]
    day = to_local(d.ts, d.cfg.local_tz).floor("D")
    worn_h = pd.Series(1, index=day).groupby(level=0).sum() * d.cfg.pre.window_s / 3600
    ax.bar(worn_h.index, worn_h.values, width=0.9, color="#9bb7d4", label="hours worn")
    for s in to_local(d.seizures, d.cfg.local_tz):
        ax.axvline(s, color="#d62728", alpha=0.25, lw=1)
    ev = to_local(d.events, d.cfg.local_tz)
    ax.plot(ev, np.full(len(ev), 25), "v", color="#d62728", label="event (lead seizure)")
    ax.set_ylim(0, 26)
    ax.set_ylabel("hours worn / day")
    ax.set_title(f"Patient {d.pid}: wear time and seizures")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_seizure_clock(d: PatientData, ax=None):
    """Polar histogram of event times of day (local)."""
    if ax is None:
        ax = plt.subplot(projection="polar")
    h = to_local(d.events, d.cfg.local_tz)
    hrs = h.hour + h.minute / 60
    th = 2 * np.pi * hrs / 24
    ax.hist(th, bins=24, color="#d62728", alpha=0.7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(2 * np.pi * np.arange(0, 24, 3) / 24)
    ax.set_xticklabels([f"{x:02d}h" for x in range(0, 24, 3)])
    ax.set_title("Events by time of day")
    return ax


def plot_cycle_scan(cycle_model, ax=None):
    """Resultant length R for every tested period; chosen cycles marked."""
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    if not cycle_model.tested:
        ax.text(0.5, 0.5, "not enough events to test cycles", ha="center", transform=ax.transAxes)
        return ax
    t = pd.DataFrame(cycle_model.tested)
    md = t[~t.local]
    ax.plot(md.period_h / 24, md.R, "o-", ms=3, label="multiday periods")
    for _, r in t[t.local].iterrows():
        ax.plot(r.period_h / 24, r.R, "s", label=f"{r.period_h:.0f} h (local time)")
    for c in cycle_model.cycles:
        ax.axvline(c.period_h / 24, color="#d62728", ls="--", alpha=0.6)
    ax.set_xlabel("period (days)")
    ax.set_ylabel("resultant length R")
    ax.set_title("Cycle scan (red = significant cycles kept)")
    ax.legend(fontsize=8)
    return ax


def plot_forecast(pred: pd.DataFrame, d: PatientData, horizon: int = 24, ax=None,
                  models=("cycle", "epicare")):
    """Forecast over time with event onsets."""
    ax = ax or plt.subplots(figsize=(12, 3))[1]
    t = pred["time"].dt.tz_convert(d.cfg.local_tz)
    for m in models:
        col = f"{m}_{horizon}h"
        if col in pred:
            ax.plot(t, pred[col], lw=1, label=m)
    ev = to_local(d.events[(d.events >= pred["time"].iloc[0].timestamp()) &
                           (d.events <= pred["time"].iloc[-1].timestamp())], d.cfg.local_tz)
    for e in ev:
        ax.axvline(e, color="#d62728", alpha=0.5, lw=1)
    ax.set_ylabel(f"P(event in {horizon} h)")
    ax.set_title(f"Patient {d.pid}: {horizon} h forecast (red lines = events)")
    ax.legend(fontsize=8)
    return ax


def plot_sen_tiw(pred: pd.DataFrame, horizon: int = 1, models=("base", "cycle", "wear", "prior_shift", "epicare"),
                 mask=None, ax=None):
    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    y = pred[f"y_{horizon}h"].to_numpy()
    m = np.ones(len(y), bool) if mask is None else mask
    for name in models:
        col = f"{name}_{horizon}h"
        if col not in pred:
            continue
        auc, tiw, sen = auc_sen_tiw(pred[col].to_numpy()[m], y[m])
        ax.plot(tiw, sen, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("time in warning")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"Sensitivity vs time in warning ({horizon} h)")
    ax.legend(fontsize=8)
    return ax


def plot_reliability(p, y, n_bins: int = 5, ax=None, label="model"):
    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    tab = reliability_table(p, y, n_bins)
    lim = max(tab.p_mean.max(), tab.observed.max()) * 1.1 + 1e-3
    ax.plot([0, lim], [0, lim], "k:", lw=0.8)
    ax.plot(tab.p_mean, tab.observed, "o-", label=label)
    for _, r in tab.iterrows():
        ax.annotate(int(r.n), (r.p_mean, r.observed), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8)
    return ax


def plot_events_seen(curve: pd.DataFrame, horizon: int = 1, metric: str = "AUC_TiW", ax=None):
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    c = curve[curve.horizon_h == horizon].copy()
    c["x"] = c.events_seen.astype(str)
    order = list(dict.fromkeys(c["x"]))
    for m, g in c.groupby("model"):
        g = g.groupby("x")[metric].mean().reindex(order)
        ax.plot(order, g.values, "o-", label=m)
    ax.axhline(0.5 if metric == "AUC_TiW" else 0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("events seen in training")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs events seen ({horizon} h)")
    ax.legend(fontsize=8)
    return ax


def plot_risk_card(out: dict, ax=None):
    """What the patient app would show for one hourly output (§4.5, §6.3)."""
    ax = ax or plt.subplots(figsize=(8, 3))[1]
    prof = np.array(out["profile_24h"])
    color = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}[out["level"]]
    ax.bar(range(24), prof, color=color, alpha=0.8)
    ax.set_xlabel("hours from now")
    ax.set_ylabel("P(event in that hour)")
    ax.set_title(f"{out['time']}  |  {out['level'].upper()} risk  |  {out['probability']:.0%} in 24 h "
                 f"(usual {out['baseline_probability']:.0%})\n{out['message']}  |  "
                 f"{', '.join(out['drivers'][:3]) or 'no specific driver'}", fontsize=9)
    return ax
