"""Personal cycle model (guidelines §4.3): seizure times only.

1. For every candidate period (12 h, 24 h, 3..45 days) the phase of each event is computed and
   the Rayleigh test checks whether the phases cluster (circadian periods use *local* time).
2. Significant periods: local maxima of the resultant length over the multiday grid, Rayleigh
   p < alpha / n_tested (Bonferroni), low-order harmonics (x2, x3) of a stronger cycle of the
   same kind removed (a daily cycle never hides a multiday one), at least ``min_full_periods`` full
   periods in the training span, at most ``max_cycles`` kept.
3. Each kept cycle gets a von Mises fit (mu, kappa) on the event phases. Hours are uniform in
   phase, so the likelihood ratio of an event at phase theta is LR = 2*pi*vM(theta; mu, kappa).
4. Event rate lambda(t) = base_rate * c * (prod_i LR_i)^(1/n_cycles) (geometric mean, as SeFEF,
   to avoid over-confidence from treating cycles as independent); c renormalises so that the
   mean rate over the training period equals the observed rate.
5. Probabilities come from a Poisson process:
   P(event in [t, t+H)) = 1 - exp(-sum over the H hours of lambda).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import i0e
from scipy.stats import vonmises

from ..config import CycleConfig
from ..seizures import to_local


def rayleigh(theta: np.ndarray) -> tuple[float, float]:
    """Mean resultant length R and Rayleigh test p-value (Zar's approximation)."""
    n = len(theta)
    if n < 2:
        return 0.0, 1.0
    C, S = np.cos(theta).sum(), np.sin(theta).sum()
    Rn = np.hypot(C, S)
    R = Rn / n
    z = Rn ** 2 / n
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n))
    return float(R), float(min(max(p, 0.0), 1.0)) if np.isfinite(z) else 1.0


def kappa_mle(R: float, n: int) -> float:
    """Best & Fisher approximation of the von Mises concentration, with small-sample correction."""
    if R < 0.53:
        k = 2 * R + R ** 3 + 5 * R ** 5 / 6
    elif R < 0.85:
        k = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        k = 1 / (R ** 3 - 4 * R ** 2 + 3 * R + 1e-9)
    if n < 15:  # bias correction (Fisher 1993)
        k = max(k - 2 / (n * k + 1e-9), 0.0) if k < 2 else (n - 1) ** 3 * k / (n ** 3 + n)
    return float(min(k, 50.0))


@dataclass
class Cycle:
    period_h: float
    mu: float
    kappa: float
    R: float
    p: float
    n_events: int
    local_time: bool

    def phase(self, t: np.ndarray, tz: str) -> np.ndarray:
        if self.local_time:
            loc = to_local(t, tz)
            secs = (loc.tz_localize(None).asi8 // 10 ** 9).astype(np.float64)  # wall-clock seconds
        else:
            secs = np.asarray(t, dtype=np.float64)
        return 2 * np.pi * np.mod(secs / 3600.0, self.period_h) / self.period_h

    def log_lr(self, t: np.ndarray, tz: str) -> np.ndarray:
        th = self.phase(t, tz)
        # log(2*pi*vM) = kappa*cos(th - mu) - log(I0(kappa)), with I0 computed stably
        return self.kappa * (np.cos(th - self.mu) - 1.0) - np.log(i0e(self.kappa))

    @property
    def name(self) -> str:
        return f"{self.period_h:.0f} h" if self.period_h < 48 else f"{self.period_h / 24:.1f} d"

    def peak_description(self, tz: str) -> str:
        frac = (self.mu % (2 * np.pi)) / (2 * np.pi)
        if self.local_time and abs(self.period_h - 24) < 1e-6:
            return f"daily cycle, peak around {int(frac * 24):02d}:00 local time"
        return f"{self.name} cycle, peak at phase {frac:.2f}"


@dataclass
class CycleModel:
    cfg: CycleConfig = field(default_factory=CycleConfig)
    tz: str = "America/Chicago"
    cycles: list[Cycle] = field(default_factory=list)
    base_rate_h: float = 0.0       # events per hour
    norm_c: float = 1.0
    tested: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ fit
    def fit(self, events: np.ndarray, t_start: float, t_end: float) -> "CycleModel":
        """Fit on events in [t_start, t_end) — the training period of one patient."""
        ev = np.sort(events[(events >= t_start) & (events < t_end)])
        hours = max((t_end - t_start) / 3600.0, 1.0)
        self.base_rate_h = (len(ev) + 0.5) / (hours + 1.0)   # light smoothing, never exactly 0
        self.cycles, self.tested = [], []
        span_h = t_end - t_start
        span_h /= 3600.0
        if len(ev) < self.cfg.min_events:
            return self
        cands = [(p, True) for p in self.cfg.circadian_periods_h]
        d = np.arange(self.cfg.multiday_min_d, self.cfg.multiday_max_d + 1e-9, 1.0)
        cands += [(float(x) * 24.0, False) for x in d]
        cands = [(p, loc) for p, loc in cands if p * self.cfg.min_full_periods <= span_h]
        if not cands:
            return self
        rows = []
        for p, loc in cands:
            c = Cycle(p, 0.0, 0.0, 0.0, 1.0, len(ev), loc)
            th = c.phase(ev, self.tz)
            R, pv = rayleigh(th)
            rows.append({"period_h": p, "local": loc, "R": R, "p": pv,
                         "mu": float(np.angle(np.exp(1j * th).mean()))})
        self.tested = rows
        # local maxima over the multiday grid (circadian candidates are always tested)
        md = [r for r in rows if not r["local"]]
        keep = [r for r in rows if r["local"]]
        for i, r in enumerate(md):
            left = md[i - 1]["R"] if i > 0 else -1
            right = md[i + 1]["R"] if i + 1 < len(md) else -1
            if r["R"] >= left and r["R"] >= right:
                keep.append(r)
        n_tested = max(len(keep), 1)
        sig = sorted([r for r in keep if r["p"] < self.cfg.alpha / n_tested], key=lambda r: r["p"])
        chosen: list[dict] = []
        for r in sig:  # remove harmonics of stronger cycles
            harmonic = False
            for c in chosen:
                # A daily (clock-time) cycle and a multiday cycle are different rhythms: every
                # whole-day period is a multiple of 24 h, so comparing them would always drop one.
                if r["local"] != c["local"]:
                    continue
                big, small = max(r["period_h"], c["period_h"]), min(r["period_h"], c["period_h"])
                ratio = big / small
                n = round(ratio)
                # only low orders (x2, x3): a broad cycle leaks into its half and third, not beyond
                if n <= self.cfg.max_harmonic_order and abs(ratio - n) < self.cfg.harmonic_tol * n:
                    harmonic = True
                    break
            if not harmonic:
                chosen.append(r)
            if len(chosen) >= self.cfg.max_cycles:
                break
        for r in chosen:
            self.cycles.append(Cycle(r["period_h"], r["mu"], kappa_mle(r["R"], len(ev)),
                                     r["R"], r["p"], len(ev), r["local"]))
        # renormalise so the mean multiplier over the training hours is 1
        if self.cycles:
            grid = np.arange(t_start, t_end, 3600.0)
            self.norm_c = 1.0 / float(np.mean(np.exp(self._log_mult(grid))))
        return self

    # ------------------------------------------------------------------ predict
    def _log_mult(self, t: np.ndarray) -> np.ndarray:
        if not self.cycles:
            return np.zeros(len(t))
        return np.mean([c.log_lr(t, self.tz) for c in self.cycles], axis=0)

    def rate(self, t: np.ndarray) -> np.ndarray:
        """Expected events per hour for the hour starting at each t."""
        t = np.asarray(t, dtype=np.float64)
        return self.base_rate_h * self.norm_c * np.exp(self._log_mult(t))

    def predict_proba(self, t: np.ndarray, horizon_h: int = 1) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if horizon_h == 1:
            lam = self.rate(t)
        else:
            offs = np.arange(horizon_h) * 3600.0
            lam = self.rate((t[:, None] + offs[None, :]).ravel()).reshape(len(t), horizon_h).sum(axis=1)
        return 1.0 - np.exp(-lam)

    def drivers(self, t: float, threshold: float = 1.3) -> list[str]:
        """Cycles whose current phase raises the risk by more than ``threshold`` x."""
        out = []
        for c in self.cycles:
            lr = float(np.exp(c.log_lr(np.array([t]), self.tz))[0])
            if lr >= threshold:
                out.append(f"{c.name} cycle phase (x{lr:.1f})")
        return out

    def summary(self) -> dict:
        return {
            "base_rate_per_day": round(self.base_rate_h * 24, 3),
            "cycles": [{"period": c.name, "R": round(c.R, 3), "p": float(f"{c.p:.2g}"),
                        "kappa": round(c.kappa, 2), "peak": c.peak_description(self.tz)}
                       for c in self.cycles],
        }
