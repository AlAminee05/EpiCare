# EpiCare mobile app

Installable web app (PWA) that shows the EpiCare results on iPhone and Android. No app store,
no build step, no dependency: plain HTML/CSS/JS that works offline once opened.

```
app/
├── index.html              shell (header, 5 tabs, bottom sheet)
├── styles.css              design system (light + dark mode)
├── app.js                  data loading, branch merge, screens, SVG charts
├── manifest.webmanifest    name, colours, icons (install)
├── sw.js                   service worker (offline shell, data network-first)
├── icons/                  app icon (any + maskable), apple-touch-icon, favicon
└── data/app_data.json      results read by the app (written by Branch 1)
```

## Screens

| Tab | Content |
|---|---|
| Aujourd'hui | 24 h risk gauge with the level thresholds, level (icon + label), plain message, next-hour risk, usual risk, confidence, hour-by-hour profile, reasons, value sent to the detector |
| Tendances | out-of-sample forecasts over the last 14 days with the seizures that happened, the thresholds and non-worn hours; signal sent to Branch 2 over 7 days |
| Cycles | recording summary, cycles found (period, R, p, peak), seizure clock |
| Fiabilité | AUC per model against chance and the best published SeFEF result, BSS, false alarms/day, calibration, events-seen curve; all patients or the selected one |
| Détection | Branch 2: placeholder now; alerts, sensitivity and false alarms once its block exists |

Every chart responds to touch (tooltip) and the main ones have a table view.

## Update the data

```bash
cd branch1
python -m epicare_b1 export-app                      # real data in data/processed -> ../app/data/app_data.json
python -m epicare_b1 export-app --events-seen        # also the events-seen experiment (slower)
python -m epicare_b1 synth && python -m epicare_b1 preprocess data/synthetic/*.zip --out data/processed_synth
python -m epicare_b1 export-app --processed data/processed_synth --synthetic   # demo data (shown with a warning banner)
```

The bundled `data/app_data.json` comes from the **synthetic** archives: the app says so in a banner.
Run `export-app` on `Mayo_1869` (and the other patients) to show real results. A results file
can also be imported on the phone itself: Réglages → Importer des résultats.

## Open it on a phone

**Quick test on the same Wi-Fi**

```bash
cd app
python -m http.server 8000
```

Find the PC's local IP (`ipconfig`, e.g. 192.168.1.20) and open `http://192.168.1.20:8000` on the phone.

**Permanent install (HTTPS, recommended)**: publish the `app/` folder on any static HTTPS host,
for example GitHub Pages (Settings → Pages → branch `main`, then open `https://<user>.github.io/EpiCare/app/`).

* **iPhone** (Safari): Share button → *Sur l'écran d'accueil*. It opens full screen with its icon.
* **Android** (Chrome): menu ⋮ → *Installer l'application* (or *Ajouter à l'écran d'accueil*).

Offline use and the Android install prompt need HTTPS (browsers only allow service workers on
HTTPS or localhost). Over plain `http://` on the local network the app still works, online.

## Merging Branch 2

The JSON is shared by all branches:

```json
{
  "schema": "epicare.app/1",
  "generated": "2026-11-25T21:30:00Z",
  "dataset": {"name": "...", "synthetic": false, "patients": ["1869"]},
  "branches": {"branch1": { ... }, "branch2": null}
}
```

Branch 2 fills `branches.branch2`, either in the same file (`export_app_data` keeps blocks it does
not own) or in a separate `data/branch2.json`, which the app merges at start-up. Fields the
Détection tab reads (others are ignored, so the block can grow freely):

```json
{
  "title": "Convulsive seizure detection",
  "metrics": {"sensitivity": 0.9, "fa_per_day": 0.12},
  "patients": [{
    "id": "1869",
    "metrics": {"sensitivity": 0.92, "fa_per_day": 0.10},
    "detections": [
      {"time": "2020-03-04T14:22", "confidence": 0.94, "prior_shift_applied": 0.8, "true_seizure": true}
    ]
  }]
}
```

Times are local wall-clock time of the patient (`YYYY-MM-DDTHH:MM`), as in Branch 1. Patients are
matched on `id`. When Branch 2 is present, the header reads "Prévision et détection", the detector
card on the home screen shows "Connecté" and the pipeline diagram lights up.

Branch 1 provides what Branch 2 needs for the fusion: `current.prior_shift_for_detector` and the
hourly series `prior_shift.values` (= clip(log(r / r̄), ±1), guidelines §5.5).
