"""Fast unit tests (no data download):  python tests/test_branch1.py   (or: python -m pytest tests -q)"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from epicare_b1.config import Config
from epicare_b1.evaluation import auc_sen_tiw, brier_skill, false_alarms_per_day
from epicare_b1.features import N_FEATURES, window_features
from epicare_b1.io_msg import parse_e4_csv, parse_seizure_times
from epicare_b1.models.cycles import CycleModel, rayleigh
from epicare_b1.models.fusion import prior_shift_fusion
from epicare_b1.seizures import group_events, hours_since_last


def test_seizure_parsing_unsorted_and_ms():
    raw = "1580000100\n1580000000\n\n1580000000000\nheader\n"
    t = parse_seizure_times(raw)
    assert list(t) == [1580000000.0, 1580000100.0]  # sorted, ms converted, duplicates removed


def test_group_events_4h_rule():
    h = 3600
    s = np.array([0, 1 * h, 3.9 * h, 8 * h, 8.5 * h, 20 * h])
    ev, ids = group_events(s, 4.0)
    assert list(ev) == [0, 8 * h, 20 * h]
    assert list(ids) == [0, 0, 0, 1, 1, 2]


def test_hours_since_last():
    out = hours_since_last(np.array([0.0, 7200.0]), np.array([3600.0]))
    assert out[0] == 24 * 30 and out[1] == 1.0


def test_e4_csv_parser():
    acc = b"1600000000.0,1600000000.0,1600000000.0\n32.0,32.0,32.0\n0,0,64\n64,0,0\n"
    s = parse_e4_csv(acc, "ACC")
    assert s.t0 == 1600000000 and s.fs == 32 and s.values.shape == (2, 3)
    assert np.allclose(np.linalg.norm(s.values, axis=1), 1.0)  # 1/64 g units -> g
    hr = parse_e4_csv(b"1600000010.0\n1.0\n70\n71\n", "HR")
    assert hr.values.tolist() == [70, 71]


def test_features_shape_and_nan_tolerance():
    rng = np.random.default_rng(0)
    acc, bvp = rng.normal(1, 0.1, (3, 1920)), rng.normal(0, 30, (3, 3840))
    eda, hr = 2 + np.abs(rng.normal(0, 0.05, (3, 240))), rng.normal(70, 2, (3, 60))
    eda[1, :50] = np.nan
    f = window_features(acc, bvp, eda, hr, eda_fs=4)
    assert f.shape == (3, N_FEATURES) == (3, 36)
    assert np.isfinite(f).all()


def test_rayleigh_detects_concentration():
    rng = np.random.default_rng(1)
    R_u, p_u = rayleigh(rng.uniform(0, 2 * np.pi, 50))
    R_c, p_c = rayleigh(rng.vonmises(0, 3, 50))
    assert p_c < 1e-4
    assert R_c > R_u


def test_cycle_model_finds_planted_multiday_cycle():
    rng = np.random.default_rng(2)
    period = 9 * 86400
    t0 = 1.6e9
    ev = np.sort(t0 + rng.integers(0, 15, 40) * period + rng.normal(0, 0.06 * period, 40))
    m = CycleModel(Config().cycle).fit(ev, t0 - period, t0 + 16 * period)
    assert any(abs(c.period_h / 24 - 9) <= 1 for c in m.cycles)
    # mean predicted hourly rate over the training span ~ observed rate (normalisation)
    grid = np.arange(t0 - period, t0 + 16 * period, 3600.0)
    assert np.isclose(m.rate(grid).mean(), m.base_rate_h, rtol=0.02)


def test_cycle_model_keeps_daily_and_multiday_cycles():
    # 7-day cycle + daily cycle peaking at 15 h: 7 d = 7 x 24 h must not count as a harmonic
    rng = np.random.default_rng(3)
    t0 = 1.6e9 - 1.6e9 % 86400                      # midnight UTC
    n = 60
    day = rng.integers(0, 15, n) * 7 + np.round(rng.normal(3, 0.8, n))
    ev = np.sort(t0 + day * 86400 + rng.normal(15, 1.5, n) * 3600)
    m = CycleModel(Config().cycle, "UTC").fit(ev, t0, t0 + 106 * 86400)
    periods = sorted(round(c.period_h) for c in m.cycles)
    assert 24 in periods and any(abs(p / 24 - 7) <= 0.5 for p in periods), periods


def test_prior_shift_is_identity_when_cycle_equals_prior():
    p = np.array([0.01, 0.1, 0.5])
    assert np.allclose(prior_shift_fusion(p, 0.05, np.full(3, 0.05)), p)


def test_metrics():
    y = np.array([0, 0, 1, 0, 1])
    perfect = np.array([0.1, 0.1, 0.9, 0.1, 0.9])
    assert auc_sen_tiw(perfect, y)[0] > 0.75
    assert np.isclose(auc_sen_tiw(np.full(5, 0.3), y)[0], 0.5)
    assert np.isclose(brier_skill(np.full(5, 0.4), y, 0.4), 0.0)
    sen, tiw, fa = false_alarms_per_day(np.array([0.9, 0.9, 0.1, 0.9, 0.1]), np.array([0, 0, 0, 1, 0]), 0.5, 1)
    assert sen == 1 and fa == 1 and tiw == 0.6


def test_synthetic_archive_roundtrip(tmp_path):
    from epicare_b1.dataset import PatientData
    from epicare_b1.forecaster import PatientForecaster
    from epicare_b1.preprocessing import process_archive, load_processed
    from epicare_b1.synthetic import SynthConfig, make_archive

    a = make_archive(tmp_path, SynthConfig(patient_id="9100", n_days=12, events_per_day=1.5,
                                           session_h=(3, 4), p_day_missing=0), verbose=False)
    w = load_processed(process_archive(a, out_dir=tmp_path / "proc", verbose=False))
    assert w["feats"].shape[1] == 36 and w["raw"].shape[1:] == (6, 60)
    assert np.all(np.diff(w["ts"]) > 0)
    cfg = Config()
    d = PatientData("9100", w, cfg)
    d.fit_normalizer(d.hours[len(d.hours) // 2])
    X, _ = d.hourly_features()
    f = PatientForecaster(cfg).fit(d, d.hours[len(d.hours) // 2], X)
    out = f.hourly_output(d, len(d.hours) - 1, X)
    assert 0 <= out["probability"] <= 1 and -1 <= out["prior_shift_for_detector"] <= 1
    assert len(out["profile_24h"]) == 24


if __name__ == "__main__":
    import inspect
    import warnings
    warnings.filterwarnings("ignore")
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
