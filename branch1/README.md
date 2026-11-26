# EpiCare: Branch 1, seizure risk forecasting

Calibrated probability of a seizure in the **next hour** and the **next 24 hours**, from
wearable data (Empatica E4) and seizure history, as specified in `ATSN.pdf` §3–4.
Dataset: *My Seizure Gauge Long-term Wearable Data* (Mayo Clinic, Zenodo
[10.5281/zenodo.17380899](https://doi.org/10.5281/zenodo.17380899), CC BY 4.0).

```
branch1/
├── epicare_b1/                 the package
│   ├── config.py               every setting from the guidelines in one place (+ Zenodo md5s)
│   ├── io_msg.py               download, archive reader (zip or folder), E4 CSV parser, inspector
│   ├── seizures.py             seizure times -> events (4 h rule), time helpers
│   ├── features.py             the 36 SeFEF hand-made features
│   ├── preprocessing.py        sessions -> 60 s windows, wear mask, filters  -> windows.npz
│   ├── dataset.py              hourly grid, labels, hour-of-day normalisation, hourly features
│   ├── models/
│   │   ├── cycles.py           personal cycle model (Rayleigh scan + von Mises + Poisson risk)
│   │   ├── shared_net.py       shared network: 1D CNN -> 5-min tokens -> GRU, masked pretraining
│   │   └── fusion.py           personal layer, SeFEF prior shift, learned fusion, calibration
│   ├── forecaster.py           PatientForecaster: fits all parts, app JSON output (§4.5)
│   ├── evaluation.py           time-series CV, AUC(Sen,TiW), BSS, reliability, FA/day, events-seen
│   ├── pipeline.py             high-level functions (load, train shared net, run, LOPO, save)
│   ├── plots.py                figures used in the notebook
│   ├── sefef_export.py         SeFEF HDF5 input file (needs h5py)
│   ├── app_export.py           results -> JSON for the mobile app (../app)
│   ├── synthetic.py            fake archives in the real layout, to test without downloading
│   └── __main__.py             command line
├── notebooks/branch1_walkthrough.ipynb    step-by-step use of everything above
├── tests/test_branch1.py       11 quick tests:  python tests/test_branch1.py
└── data/ models/ results/      created on use, ignored by git
```

## Quick start

```bash
pip install -r requirements.txt
python tests/test_branch1.py                 # 11 tests, ~15 s
jupyter notebook notebooks/branch1_walkthrough.ipynb   # SYNTHETIC = False: real data in data/raw/; True: fake data, no download
```

Command line, same steps:

```bash
python -m epicare_b1 synth                              # 3 fake patients in data/synthetic
python -m epicare_b1 download 1869                      # 3.3 GB, resumable, md5-checked
python -m epicare_b1 inspect data/raw/Mayo_1869.zip     # check the format first
python -m epicare_b1 preprocess data/raw/Mayo_1869.zip  # -> data/processed/1869/windows.npz
python -m epicare_b1 evaluate --representation features # or: shared
python -m epicare_b1 export-app                         # -> ../app/data/app_data.json (mobile app)
```

## What the dataset looks like (confirmed on Mayo_1869)

* 8 archives `Mayo_1110/1869/1876/1904/1927/1965/1988/2002.zip`, 3.3–6.9 GB, plus
  `SeizureTimesOnly.zip` (1.3 kB: `1219.txt`, `1675.txt`, `1942.txt`, patients without wearable data).
* Inside `Mayo_1869.zip`: a folder `Mayo_1869/` with `1869.txt` and 182 nested zips named
  `<unix start>_<device>.zip`, e.g. `1577897948_A025CF.zip`. Devices: `A029E3`, `A025CF`, `A029B4`
  (E4 serial numbers) and `000000` (3 sessions, also E4 data, serial missing).
* Each nested zip is an **Empatica E4 Connect export**: `ACC.csv` (32 Hz, x/y/z in 1/64 g, ±2 g),
  `BVP.csv` (64 Hz), `EDA.csv` (4 Hz), `TEMP.csv` (4 Hz), `HR.csv` (1 Hz), `IBI.csv`, `tags.csv`,
  `info.txt`. Each CSV: row 1 = start (Unix s, UTC), row 2 = rate (Hz), then samples.
  Sessions last about a day; two watches alternate while one charges, so wear is close to 24/7.
* `<id>.txt` = seizure onsets, one Unix timestamp per line, onset only (26 onsets for 1869,
  2020-01-21 to 2020-07-23).
* The loader reads start and rate from each file rather than hard-coding them, so the other
  archives should load as they are; run `inspect` on each new archive anyway.

Put archives in `data/raw/` (the `download` command does this). An extracted copy is not needed:
the loader reads the nested zips straight from the outer zip.

## Method (summary: see the notebook for details)

| Part | Implementation |
|---|---|
| Events | seizure starts a new event after ≥ 4 h without seizures; labels use events only |
| Wear mask | window kept if ACC, BVP, EDA, HR each ≥ 80 % present, skin temp ≥ 25 °C, EDA ≥ 0.02 µS; gaps = missing |
| Normalisation | per patient, per local hour of day, robust z-score fitted on the training part only |
| Cycle model | 12 h and 24 h (local time) and 3–45 d; ≥ 3 full periods; Rayleigh + Bonferroni; ×2 and ×3 harmonics removed within daily or within multiday (a daily cycle never hides a multiday one); von Mises per cycle; Poisson rate (geometric mean of cycle likelihood ratios, renormalised) |
| Shared network | 1D CNN per 60 s window (6 ch @1 Hz + 36 features), masked mean into 5-min tokens with a learned *missing* token, bi-GRU over 4 h, 32-d hour embedding; masked-token reconstruction pretraining, then a light supervised step |
| Light fallback | hourly means of the 36 SeFEF features over the last 1 h, 6 h and 24 h |
| Personal layer | L2 logistic regression (C = 0.1) per patient, used from 10 events onwards |
| Fusion | baseline = SeFEF prior shift; ours = logistic regression on [logit wear, logit cycle, log(1 + h since last seizure)] fitted on the latest 30 % of training (also calibrates) |
| Outputs | P(1 h), P(24 h), level (per-patient quantiles), plain message, 24 h profile, drivers, confidence, `prior_shift_for_detector = clip(log(r/r̄), ±1)` |
| Evaluation | expanding-window TSCV (≥ 3 events in train and test), AUC(Sen, TiW), BSS vs training climatology, reliability, Sen/TiW/FA per day at a threshold set on training; events-seen curve; LOPO for the shared network |
