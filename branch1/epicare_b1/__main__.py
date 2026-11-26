"""Command line:  python -m epicare_b1 <command> ...

  synth       make synthetic MSG-format archives (test the pipeline without downloading)
  download    download patient archives from Zenodo (md5-checked, resumable)
  inspect     list what is inside an archive (run first on Mayo_1869, the smallest)
  preprocess  archive(s) -> data/processed/<id>/windows.npz
  evaluate    time-series CV for all processed patients (features or shared network)
  export-app  run the CV and write the JSON read by the mobile app (../app/data/app_data.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config, ZENODO_FILES


def main(argv=None):
    ap = argparse.ArgumentParser(prog="epicare_b1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--ids", nargs="+", default=["9001", "9002", "9003"])
    s.add_argument("--days", type=int, default=60)

    s = sub.add_parser("download")
    s.add_argument("ids", nargs="+", help=f"patient ids, among {list(ZENODO_FILES)}")
    s.add_argument("--out", default="data/raw")

    s = sub.add_parser("inspect")
    s.add_argument("archive")

    s = sub.add_parser("preprocess")
    s.add_argument("archives", nargs="+")
    s.add_argument("--out", default="data/processed")
    s.add_argument("--max-sessions", type=int, default=None)

    s = sub.add_parser("evaluate")
    s.add_argument("--processed", default="data/processed")
    s.add_argument("--ids", nargs="*", default=None)
    s.add_argument("--representation", choices=["features", "shared", "none"], default="features")
    s.add_argument("--out", default="results")

    s = sub.add_parser("export-app", help="results -> JSON for the mobile app (app/)")
    s.add_argument("--processed", default="data/processed")
    s.add_argument("--ids", nargs="*", default=None)
    s.add_argument("--out", default="../app/data/app_data.json")
    s.add_argument("--synthetic", action="store_true", help="mark the data as synthetic in the app")
    s.add_argument("--events-seen", action="store_true", help="also run the events-seen experiment")

    a = ap.parse_args(argv)
    cfg = Config()
    if a.cmd == "synth":
        from .synthetic import SynthConfig, make_archive
        for i, pid in enumerate(a.ids):
            make_archive(a.out, SynthConfig(patient_id=pid, n_days=a.days, seed=i,
                                            peak_hour=[15, 11, 18, 9][i % 4],
                                            multiday_period_d=[7, 10, 5, 13][i % 4]))
    elif a.cmd == "download":
        from .io_msg import download_archive
        for pid in a.ids:
            print(download_archive(pid, Path(a.out)))
    elif a.cmd == "inspect":
        from .io_msg import inspect_archive
        inspect_archive(a.archive)
    elif a.cmd == "preprocess":
        from .preprocessing import process_archive
        for arc in a.archives:
            process_archive(arc, cfg, a.out, max_sessions=a.max_sessions)
    elif a.cmd == "evaluate":
        from .pipeline import compare_with_sefef, load_patients, run_branch1, train_shared_network
        P = load_patients(a.processed, a.ids, cfg)
        model = batcher = None
        if a.representation == "shared":
            model, batcher, _ = train_shared_network(P, cfg)
        res, summ, _ = run_branch1(P, cfg, a.representation, model, batcher)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / f"folds_{a.representation}.csv", index=False)
        summ.to_csv(out / f"summary_{a.representation}.csv")
        print(summ.to_string())
        print(compare_with_sefef(summ).to_string(index=False))
        (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1, default=str))
    elif a.cmd == "export-app":
        import pandas as pd
        from .app_export import export_app_data
        from .evaluation import events_seen_curve
        from .pipeline import load_patients, run_branch1
        P = load_patients(a.processed, a.ids, cfg)
        res, _, fitted = run_branch1(P, cfg, "features")
        curve = None
        if a.events_seen:
            days = 20 if a.synthetic else 60
            curve = pd.concat([events_seen_curve(d, cfg, test_days=days) for d in P.values()],
                              ignore_index=True)
        print(export_app_data(P, res, fitted, cfg, a.out, synthetic=a.synthetic, events_seen=curve))


if __name__ == "__main__":
    main()
