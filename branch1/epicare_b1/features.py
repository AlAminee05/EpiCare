"""The 36 hand-made window features of the SeFEF paper (Carmo et al. 2025, §4.1 fallback).

* 28 statistical features = 7 statistics x 4 modalities (ACC magnitude, BVP, EDA, HR):
  mean, power, std, kurtosis, skewness, mean |1st difference|, mean |2nd difference|
* 3 Hjorth parameters on EDA: activity, mobility, complexity
* 5 EDA event features: number of peaks, mean amplitude, mean rise time,
  sum of amplitudes, sum of rise times

All functions work on a matrix ``X`` of shape (n_windows, samples_per_window) that may contain
NaN for missing samples, so a whole session is processed at once.

Note: the paper says "mean of first/second-order differences" without saying signed or
absolute; the signed mean is ~0 for any stationary signal, so absolute differences are used.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

STAT_NAMES = ["mean", "power", "std", "kurtosis", "skewness", "d1", "d2"]
STAT_MODALITIES = ["acc", "bvp", "eda", "hr"]
HJORTH_NAMES = ["eda_hjorth_activity", "eda_hjorth_mobility", "eda_hjorth_complexity"]
EVENT_NAMES = ["eda_n_peaks", "eda_peak_amp_mean", "eda_rise_time_mean", "eda_peak_amp_sum", "eda_rise_time_sum"]
FEATURE_NAMES = [f"{m}_{s}" for m in STAT_MODALITIES for s in STAT_NAMES] + HJORTH_NAMES + EVENT_NAMES
N_FEATURES = len(FEATURE_NAMES)  # 36


def _nanmean(x, axis=1):
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.sum(~np.isnan(x), axis=axis)
        s = np.nansum(x, axis=axis)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def statistical(X: np.ndarray) -> np.ndarray:
    """7 statistics per row -> (n, 7)."""
    m = _nanmean(X)
    c = X - m[:, None]
    m2 = _nanmean(c ** 2)
    m3 = _nanmean(c ** 3)
    m4 = _nanmean(c ** 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(m2)
        skew = np.where(m2 > 1e-12, m3 / np.power(m2, 1.5), 0.0)
        kurt = np.where(m2 > 1e-12, m4 / m2 ** 2 - 3.0, 0.0)
    power = _nanmean(X ** 2)
    d1 = _nanmean(np.abs(np.diff(X, axis=1)))
    d2 = _nanmean(np.abs(np.diff(X, n=2, axis=1)))
    return np.stack([m, power, std, kurt, skew, d1, d2], axis=1)


def hjorth(X: np.ndarray) -> np.ndarray:
    """Hjorth activity, mobility, complexity per row -> (n, 3)."""
    d1 = np.diff(X, axis=1)
    d2 = np.diff(X, n=2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v0 = np.nanvar(X, axis=1)
        v1 = np.nanvar(d1, axis=1)
        v2 = np.nanvar(d2, axis=1)
        mob = np.sqrt(np.where(v0 > 1e-12, v1 / v0, 0.0))
        mob_d = np.sqrt(np.where(v1 > 1e-12, v2 / v1, 0.0))
        comp = np.where(mob > 1e-12, mob_d / mob, 0.0)
    return np.stack([v0, mob, comp], axis=1)


def eda_events(X: np.ndarray, fs: float, prominence: float = 0.01) -> np.ndarray:
    """Skin-conductance response peaks per row -> (n, 5)."""
    out = np.zeros((X.shape[0], 5))
    for i, row in enumerate(X):
        row = row[~np.isnan(row)]
        if len(row) < 3:
            out[i] = np.nan
            continue
        pk, props = find_peaks(row, prominence=prominence)
        if len(pk) == 0:
            continue
        amp = props["prominences"]
        rise = (pk - props["left_bases"]) / fs
        out[i] = [len(pk), amp.mean(), rise.mean(), amp.sum(), rise.sum()]
    return out


def window_features(acc_mag: np.ndarray, bvp: np.ndarray, eda: np.ndarray, hr: np.ndarray,
                    eda_fs: float, eda_prominence: float = 0.01) -> np.ndarray:
    """All 36 features for a batch of windows -> (n, 36) float32."""
    parts = [statistical(acc_mag), statistical(bvp), statistical(eda), statistical(hr),
             hjorth(eda), eda_events(eda, eda_fs, eda_prominence)]
    return np.concatenate(parts, axis=1).astype(np.float32)
