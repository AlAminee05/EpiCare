"""Central configuration for EpiCare Branch 1 (seizure risk forecasting).

Every number that the project guidelines (ATSN.pdf) fix is collected here so it can be
changed in one place. Paths are relative to the ``branch1/`` folder by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

BRANCH1_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Dataset: My Seizure Gauge Long-term Wearable Data (Mayo Clinic), Zenodo record 17380899
# DOI 10.5281/zenodo.17380899, CC BY 4.0.  Sizes / md5 come from the Zenodo API.
# --------------------------------------------------------------------------------------
ZENODO_RECORD_ID = 17380899
ZENODO_FILES = {
    # patient id: (file name, size in bytes, md5)
    "1110": ("Mayo_1110.zip", 3_780_354_793, "d833a1979f753bb6896d01fbfba1ea53"),
    "1869": ("Mayo_1869.zip", 3_267_650_706, "3246f7f811f968370b04235234d28dee"),
    "1876": ("Mayo_1876.zip", 3_770_594_324, "331be6ae928010c8bcdc0566268d4a3e"),
    "1904": ("Mayo_1904.zip", 6_021_226_102, "e9ec4f352f03366295d348b7d8841288"),
    "1927": ("Mayo_1927.zip", 5_922_077_661, "6f835293c48ae0e33d8072abbdabbe2c"),
    "1965": ("Mayo_1965.zip", 6_888_541_242, "dbd6f19123edc774ab5496cbd56588e3"),
    "1988": ("Mayo_1988.zip", 4_948_245_809, "453a239c233df387cf71be7ee50cc220"),
    "2002": ("Mayo_2002.zip", 4_483_698_587, "752bd702bb465cd1c81a226a81d5f3a1"),
    "seizure_times_only": ("SeizureTimesOnly.zip", 1_328, "e1964dc7c7b73dc6109a149e7629a9d4"),
}
SEFEF_PATIENTS = ["1110", "1869", "1876", "1904", "1965", "2002"]  # direct baseline
EXTRA_PATIENTS = ["1927", "1988"]
LOCAL_TZ = "America/Chicago"  # Mayo Clinic, Rochester MN (US Central)

# Published SeFEF results (Carmo et al. 2025, Table 2), hourly / daily AUC(Sen, TiW).
SEFEF_BASELINE_AUC = {
    "hourly": {"vMe": 0.66, "LR": 0.50, "LR-vMe": 0.53},
    "daily": {"vMe": 0.47, "LR": 0.40, "LR-vMe": 0.41},
}

# Signals the shared network sees. Each becomes one channel of the 1 Hz "raw" window.
RAW_CHANNELS = ["acc_mean", "acc_std", "bvp_amp", "eda", "hr", "temp"]


@dataclass
class PreprocessConfig:
    window_s: int = 60                 # Branch 1 window length (guidelines §3, step 5)
    raw_fs: int = 1                    # rate of the compact raw window kept for the network
    min_coverage: float = 0.8          # fraction of expected samples a modality needs in a window
    required_modalities: tuple = ("acc", "bvp", "eda", "hr")  # SeFEF's four modalities
    # Off-wrist detection (E4 is "worn" only if skin is warm and EDA is non-zero)
    min_skin_temp_c: float = 25.0
    min_eda_us: float = 0.02
    # Filters (SeFEF Appendix B, adapted to E4 rates)
    bvp_band_hz: tuple = (1.0, 7.0)
    eda_lowpass_hz: float = 1.0        # E4 EDA is 4 Hz, so the 5 Hz SeFEF low-pass cannot apply
    eda_smooth_s: float = 0.75
    eda_peak_prominence_us: float = 0.01


@dataclass
class LabelConfig:
    event_gap_h: float = 4.0           # new event only after >= 4 h without seizures (§3, step 3)
    post_ictal_h: float = 1.0          # hours after any seizure removed from *training* targets
    horizons_h: tuple = (1, 24)        # next hour and next 24 hours (§1, §4.5)
    min_hour_coverage: float = 0.5     # fraction of the hour's windows that must be worn


@dataclass
class SharedNetConfig:
    context_h: int = 4                 # hours of history the GRU looks at
    token_min: int = 5                 # windows are pooled into 5-minute tokens before the GRU
    window_emb: int = 32
    hour_emb: int = 32
    use_features: bool = True          # also feed the 36 hand-made features of each window
    dropout: float = 0.1
    mask_ratio: float = 0.3            # masked-token pretraining
    pretrain_epochs: int = 5
    finetune_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    neg_per_pos: int = 20              # negative hours sampled per positive hour when fine-tuning
    max_hours_per_patient: int | None = 6000  # subsample hours per epoch (CPU friendly)
    device: str = "cpu"
    seed: int = 0


@dataclass
class CycleConfig:
    circadian_periods_h: tuple = (12.0, 24.0)
    multiday_min_d: float = 3.0
    multiday_max_d: float = 45.0
    min_full_periods: int = 3          # a cycle is trusted only after 3 full periods (§4.3)
    min_events: int = 5
    alpha: float = 0.05                # Rayleigh test level, Bonferroni-corrected
    max_cycles: int = 3
    harmonic_tol: float = 0.08
    max_harmonic_order: int = 3        # only x2 and x3 count as harmonics, within daily or multiday


@dataclass
class FusionConfig:
    min_events_personal: int = 10      # personal layer only after 10 events (§4.2)
    min_events_fusion: int = 10        # learned fusion only after 10 events (else prior shift)
    personal_C: float = 0.1            # strong L2: few parameters effectively learnt
    calib_frac: float = 0.3            # last 30 % of the training period fits fusion + calibration
    min_calib_events: int = 2
    calibration: str = "platt"         # "platt" | "isotonic" | "none"
    prior_shift_cap: float = 1.0       # |log(r / r_bar)| <= 1 for Branch 2 (§5.5)


@dataclass
class EvalConfig:
    n_min_events_train: int = 3
    n_min_events_test: int = 3
    initial_train_frac: float = 1 / 3
    test_frac_of_initial: float = 0.5
    daily_issue_hour_local: int = 0    # daily forecasts issued at local midnight
    n_reliability_bins: int = 5
    events_seen_grid: tuple = (0, 5, 10, 20, None)  # None = all


@dataclass
class Config:
    data_raw: Path = BRANCH1_ROOT / "data" / "raw"
    data_processed: Path = BRANCH1_ROOT / "data" / "processed"
    models_dir: Path = BRANCH1_ROOT / "models"
    results_dir: Path = BRANCH1_ROOT / "results"
    local_tz: str = LOCAL_TZ
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    net: SharedNetConfig = field(default_factory=SharedNetConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
