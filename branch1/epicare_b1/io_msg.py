"""Loaders for the My Seizure Gauge (MSG) Zenodo release.

Archive layout, confirmed on Mayo_1869 (Zenodo record 17380899):

    Mayo_1869.zip                         <- 3.3 GB, stored (not compressed)
    └── Mayo_1869/
        ├── 1869.txt                      <- seizure onsets, one Unix timestamp per line (26 lines)
        ├── 1577897948_A025CF.zip         <- 182 nested zips "<unix start>_<device serial>.zip"
        ├── 1577980964_A029E3.zip            (devices A029E3, A025CF, A029B4, 000000)
        └── ...

Every nested zip is an Empatica E4 Connect session export: ACC, BVP, EDA, HR, IBI, TEMP and
tags CSVs plus ``info.txt``. In every signal CSV the first row is the session start (Unix
seconds, UTC), the second row the sampling rate in Hz, then one sample per row (ACC: 3 columns
x/y/z in 1/64 g, range +/-2 g). Rates seen: ACC 32 Hz, BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, HR 1 Hz.
Sessions last about one day; two devices alternate (one charges while the other is worn).
The ``_000000`` sessions are ordinary E4 exports too (device serial missing), not Fitbit data.

The module stays defensive so the other archives load even if they differ:
  * it reads start time and rate from each file rather than assuming them,
  * it accepts the archive as the outer zip (no extraction needed) or as a folder,
  * :func:`inspect_archive` prints what is really inside an archive,
  * sessions without any E4 signal CSV are reported and skipped instead of crashing.
"""
from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import ZENODO_FILES, ZENODO_RECORD_ID

SESSION_RE = re.compile(r"(?P<start>\d{9,11})_(?P<device>[0-9A-Za-z]+)\.zip$")
E4_SIGNALS = ("ACC", "BVP", "EDA", "HR", "TEMP")


# ----------------------------------------------------------------------------- download
def zenodo_url(patient_id: str) -> str:
    fname = ZENODO_FILES[str(patient_id)][0]
    return f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fname}?download=1"


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_archive(patient_id: str, dest_dir: Path, verify: bool = True) -> Path:
    """Download one patient archive from Zenodo (resumable) and check its md5."""
    patient_id = str(patient_id)
    fname, size, md5 = ZENODO_FILES[patient_id]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / fname
    have = out.stat().st_size if out.exists() else 0
    if have < size:
        req = urllib.request.Request(zenodo_url(patient_id), headers={"User-Agent": "epicare-branch1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req) as r, open(out, "ab" if have else "wb") as f:
            done = have
            while True:
                block = r.read(1 << 22)
                if not block:
                    break
                f.write(block)
                done += len(block)
                print(f"\r{fname}: {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="")
        print()
    if verify:
        got = md5sum(out)
        if got != md5:
            raise IOError(f"md5 mismatch for {out}: {got} != {md5}")
    return out


# ----------------------------------------------------------------------------- archive access
class Archive:
    """Uniform access to a patient archive given as the outer zip OR an extracted folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.suffix.lower() == ".zip" else None

    def names(self) -> list[str]:
        if self._zip is not None:
            return [n for n in self._zip.namelist() if not n.endswith("/")]
        return [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def close(self):
        if self._zip is not None:
            self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def patient_id(self) -> str:
        m = re.search(r"(\d{3,5})", self.path.stem)
        return m.group(1) if m else self.path.stem


@dataclass
class SessionRef:
    name: str          # member name inside the archive
    start: int         # unix seconds from the file name
    device: str


@dataclass
class Signal:
    t0: float                       # unix seconds (UTC) of the first sample
    fs: float                       # Hz
    values: np.ndarray              # (n,) or (n, 3) for ACC

    @property
    def t_end(self) -> float:
        return self.t0 + len(self.values) / self.fs


@dataclass
class Session:
    ref: SessionRef
    signals: dict[str, Signal] = field(default_factory=dict)
    other_files: list[str] = field(default_factory=list)

    @property
    def is_e4(self) -> bool:
        return any(s in self.signals for s in E4_SIGNALS)


def list_sessions(archive: Archive) -> list[SessionRef]:
    refs = []
    for n in archive.names():
        m = SESSION_RE.search(n.split("/")[-1])
        if m:
            refs.append(SessionRef(n, int(m.group("start")), m.group("device")))
    return sorted(refs, key=lambda r: r.start)


def parse_e4_csv(raw: bytes, signal: str) -> Signal | None:
    """Parse one E4 Connect CSV: row 1 = start time(s), row 2 = rate(s), then samples."""
    text = raw.decode("utf-8", errors="replace").strip().splitlines()
    if len(text) < 3:
        return None
    try:
        t0 = float(text[0].split(",")[0])
        fs = float(text[1].split(",")[0])
    except ValueError:
        return None
    vals = pd.read_csv(io.StringIO("\n".join(text[2:])), header=None).to_numpy(dtype=np.float64)
    if signal == "ACC":
        vals = vals[:, :3] / 64.0     # E4 ACC is in 1/64 g
    else:
        vals = vals[:, 0]
    if t0 > 1e11:                      # milliseconds, just in case
        t0 /= 1000.0
    return Signal(t0=t0, fs=fs, values=vals)


def load_session(archive: Archive, ref: SessionRef) -> Session:
    sess = Session(ref=ref)
    try:
        inner = zipfile.ZipFile(io.BytesIO(archive.read(ref.name)))
    except zipfile.BadZipFile:
        warnings.warn(f"{ref.name}: not a valid zip, skipped")
        return sess
    with inner:
        for n in inner.namelist():
            base = n.split("/")[-1]
            stem = base.rsplit(".", 1)[0].upper()
            if base.lower().endswith(".csv") and stem in E4_SIGNALS:
                sig = parse_e4_csv(inner.read(n), stem)
                if sig is not None and len(sig.values):
                    sess.signals[stem] = sig
            else:
                sess.other_files.append(n)
    return sess


def iter_sessions(archive: Archive, skip_non_e4: bool = True) -> Iterator[Session]:
    for ref in list_sessions(archive):
        s = load_session(archive, ref)
        if skip_non_e4 and not s.is_e4:
            warnings.warn(f"{ref.name}: no E4 signal CSV found (files: {s.other_files[:6]}), skipped")
            continue
        yield s


# ----------------------------------------------------------------------------- seizures
def parse_seizure_times(raw: bytes | str) -> np.ndarray:
    """One onset per line, Unix seconds (ms tolerated); returns sorted unique float array."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    vals = []
    for line in raw.replace(",", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            v = float(line)
        except ValueError:
            continue  # header or comment
        vals.append(v / 1000.0 if v > 1e11 else v)
    return np.unique(np.asarray(vals, dtype=np.float64))  # np.unique also sorts


def find_seizure_file(archive: Archive) -> str | None:
    """The ``<id>.txt`` file at the top of a patient folder (e.g. ``Mayo_1869/1869.txt``)."""
    pid = archive.patient_id
    cands = [n for n in archive.names() if n.lower().endswith(".txt") and not SESSION_RE.search(n)]
    exact = [n for n in cands if n.split("/")[-1] == f"{pid}.txt"]
    return (exact or cands or [None])[0]


def load_seizure_times(archive: Archive) -> np.ndarray:
    name = find_seizure_file(archive)
    if name is None:
        raise FileNotFoundError(f"No seizure .txt file in {archive.path}")
    times = parse_seizure_times(archive.read(name))
    if len(times) == 0:
        raise ValueError(f"{name} contained no timestamps; open it and check the format")
    return times


# ----------------------------------------------------------------------------- inspection
def inspect_archive(path: str | Path, n_sessions: int = 3, n_lines: int = 4) -> dict:
    """Print (and return) what is inside an archive. Run this first on Mayo_1869."""
    report: dict = {}
    with Archive(path) as ar:
        names = ar.names()
        refs = list_sessions(ar)
        report["n_files"] = len(names)
        report["n_sessions"] = len(refs)
        report["devices"] = pd.Series([r.device for r in refs]).value_counts().to_dict()
        report["other_files"] = [n for n in names if not SESSION_RE.search(n)]
        print(f"{ar.path.name}: {len(names)} files, {len(refs)} session zips")
        print("devices:", report["devices"])
        print("non-session files:", report["other_files"])
        sz = find_seizure_file(ar)
        if sz:
            raw = ar.read(sz).decode("utf-8", errors="replace")
            print(f"\n--- {sz} (first lines) ---\n" + "\n".join(raw.splitlines()[:8]))
            t = parse_seizure_times(raw)
            report["seizures"] = len(t)
            if len(t):
                print(f"parsed {len(t)} onsets, {pd.to_datetime(t[0], unit='s', utc=True)} -> "
                      f"{pd.to_datetime(t[-1], unit='s', utc=True)}")
        # one session per device type, then the first few
        picked, seen = [], set()
        for r in refs:
            if r.device not in seen:
                picked.append(r)
                seen.add(r.device)
        picked = (picked + [r for r in refs if r not in picked])[: max(n_sessions, len(seen))]
        report["sessions"] = {}
        for r in picked:
            inner = zipfile.ZipFile(io.BytesIO(ar.read(r.name)))
            print(f"\n=== {r.name}  (start {pd.to_datetime(r.start, unit='s', utc=True)}) ===")
            info = {}
            for zi in inner.infolist():
                head = inner.read(zi.filename)[:400].decode("utf-8", errors="replace").splitlines()[:n_lines]
                info[zi.filename] = {"size": zi.file_size, "head": head}
                print(f"  {zi.filename:<14} {zi.file_size:>11,d} B | " + " | ".join(head))
            report["sessions"][r.name] = info
    return report
