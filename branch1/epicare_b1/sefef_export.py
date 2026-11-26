"""Write SeFEF's input format (guidelines §3 step 5) and optionally score with the sefef package.

SeFEF (``pip install sefef``, v3.0.0) reads an HDF5 file with:
    timestamps  int64  (n,)        window start, Unix seconds
    data        float32 (n, d)     one row per window (the guidelines call it "samples";
                                   the sefef code and docs name it ``data``, so both are written)
    annotations bool    (n,)       added by ``sefef.labeling.add_annotations``
    sz_onsets   int64   (k,)       added by ``sefef.labeling.add_sz_onsets``
Requires ``h5py`` (``pip install h5py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_sefef_h5(windows: dict, path: str | Path, use_lead_seizures: bool = True,
                   add_labels: bool = True, preictal_s: int = 3600, latency_s: int = 600) -> Path:
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onsets = (windows["events"] if use_lead_seizures else windows["seizures"]).astype(np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamps", data=windows["ts"].astype(np.int64))
        f.create_dataset("data", data=windows["feats"].astype(np.float32))
        f["samples"] = f["data"]  # hard link, same bytes under the guidelines' name
    if add_labels:
        try:
            from sefef import labeling
            with h5py.File(path, "r+") as f:
                labeling.add_annotations(f, onsets, preictal_duration=preictal_s, prediction_latency=latency_s)
                labeling.add_sz_onsets(f, onsets)
        except ImportError:
            # same labelling rule without sefef: preictal = [onset - preictal - latency, onset - latency)
            ts = windows["ts"].astype(np.int64)
            ann = np.zeros(len(ts), bool)
            for o in onsets:
                ann |= (ts >= o - preictal_s - latency_s) & (ts < o - latency_s)
            with h5py.File(path, "r+") as f:
                f.create_dataset("annotations", data=ann)
                f.create_dataset("sz_onsets", data=onsets)
    return path
