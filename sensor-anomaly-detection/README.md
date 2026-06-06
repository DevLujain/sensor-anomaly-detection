# Drone Telemetry Anomaly Detection

Time-series **anomaly detection** on real multi-drone flight telemetry. The
pipeline ingests raw mission logs (per-drone state, position, and battery
sampled at ~2 Hz), engineers motion and power features, and flags anomalous
samples using two complementary detectors:

1. **Rule-based, physics-grounded** — interpretable checks an operator would
   trust: impossible altitude readings (hard sensor faults), out-of-envelope
   altitude, implausible vertical/horizontal speed, battery jumps, and sensor
   dropouts.
2. **Isolation Forest (unsupervised)** — learns multivariate outliers across
   all features, catching anomalous *combinations* no single rule describes.

The two are combined into a final per-sample anomaly label, with full reasons
exported so every flag is explainable.

## Why this exists

This is the kind of asset-monitoring problem found in industrial IoT / track
& trace: a fleet of moving assets streaming position and status over time,
where the goal is to spot faults, glitches, and abnormal behaviour
automatically. The data here is genuine telemetry from an autonomous
multi-drone mission system (Stage-16 logs), reused as a realistic sensor
stream.

## Data

Real telemetry CSVs (`telemetry_*.csv`), wide multi-drone schema:

```
ts, day, D0_state, D0_x, D0_y, D0_z, D0_batt, D1_state, D1_x, D1_y, D1_z, D1_batt
```

`load.py` reshapes this into tidy long form (one row per timestamp per drone).
The bundled run used **48 mission files, 307,756 samples, 2 drones, 8 days**.

## Run

```bash
pip install -r requirements.txt
python -m src.run --data data/stage16_telemetry --out results
```

Options: `--contamination 0.01` (Isolation Forest outlier fraction).

## Outputs (in `results/`)

| File | Contents |
| --- | --- |
| `anomaly_summary.csv` | total samples, overall anomaly rate, per-rule counts |
| `per_drone_summary.csv` | anomaly counts and rate per drone |
| `anomalies.csv` | every flagged sample with all rule reasons + Isolation Forest score |
| `timeline_<run>_<drone>.png` | battery & altitude over time, anomalies marked |

### Example result (bundled run)

| Metric | Value |
| --- | --- |
| Total samples | 307,756 |
| Anomalies flagged | 17,549 (5.7%) |
| Hard sensor faults (impossible altitude) | 17,099 |
| Altitude envelope excursions | 202 |
| Battery jumps | 139 |
| Horizontal-speed outliers | 80 |
| Isolation Forest multivariate outliers | 3,078 |

The dominant finding — thousands of physically impossible altitude readings
(tens to thousands of metres, and large negatives) — are genuine sensor
glitches in the raw logs, exactly what an anomaly detector should surface for
data-quality triage before the telemetry is trusted downstream.

## Structure

```
src/
  load.py       # wide multi-drone CSV -> tidy long-form time series
  features.py   # speed, vertical rate, battery rate, sample-gap features
  detect.py     # rule-based + Isolation Forest detectors, combined
  run.py        # end-to-end pipeline: load -> features -> detect -> report/plots
data/
  stage16_telemetry/   # telemetry CSVs (place your files here)
results/               # summaries, flagged samples, timeline plots
requirements.txt
```

## Notes

- Rules are split by severity (`alt_fault` = impossible vs. `alt_envelope` =
  mild excursion) so hard glitches are distinguished from minor deviations.
- The Isolation Forest is unsupervised — no labels needed — which suits real
  telemetry where ground-truth fault labels rarely exist.
- All thresholds are explicit arguments in `detect.rule_flags`, easy to tune
  to a different platform's operating envelope.
