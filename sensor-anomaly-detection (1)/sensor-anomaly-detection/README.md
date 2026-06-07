# Drone Telemetry Anomaly Detection

I built this to find anomalies in real drone flight logs. The data is telemetry
from my final-year project's multi-drone system: each drone logs its state,
position (x, y, z) and battery about twice a second. The pipeline loads those
logs, derives a few motion and power features, and flags samples that look
wrong.

It uses two detectors and combines them:

1. Rule-based checks. Simple physics rules: altitude that's impossible or out of
   range, vertical or horizontal speed that's too high, sudden battery jumps,
   and gaps where a sample arrived far later than expected. These are easy to
   explain and an operator can trust them.
2. Isolation Forest. An unsupervised model that finds odd combinations across
   all the features, which the fixed rules would miss.

A sample is flagged if either the rules or the model catch it, and every flag
keeps its reasons so I can see why.

## Why I made it

It's basically an asset-monitoring problem: a few moving things streaming
position and status over time, and you want to spot faults and bad readings
without labelling everything by hand. The telemetry is my own from the drone
project, reused as a realistic sensor stream.

## Data

Telemetry CSVs (`telemetry_*.csv`) in a wide, multi-drone layout:

```
ts, day, D0_state, D0_x, D0_y, D0_z, D0_batt, D1_state, D1_x, D1_y, D1_z, D1_batt
```

`load.py` turns this into long form (one row per timestamp per drone). The run
here used 48 files: 307,756 samples, 2 drones, 8 days.

## Run

```bash
pip install -r requirements.txt
python -m src.run --data data/stage16_telemetry --out results
```

`--contamination 0.01` sets the Isolation Forest outlier fraction.

## Outputs (in `results/`)

| File | What's in it |
| --- | --- |
| `anomaly_summary.csv` | total samples, anomaly rate, count per rule |
| `per_drone_summary.csv` | anomaly counts and rate per drone |
| `anomalies.csv` | every flagged sample with its reasons + Isolation Forest score |
| `timeline_<run>_<drone>.png` | battery and altitude over time, anomalies marked |

### What the run found

| Metric | Value |
| --- | --- |
| Total samples | 307,756 |
| Anomalies flagged | 17,549 (5.7%) |
| Hard sensor faults (impossible altitude) | 17,099 |
| Altitude envelope excursions | 202 |
| Battery jumps | 139 |
| Horizontal-speed outliers | 80 |
| Isolation Forest multivariate outliers | 3,078 |

Most of the flags are impossible altitude readings — values of tens, hundreds,
even thousands of metres, plus large negatives. A sub-4-metre drone can't be at
2000 m, so these are genuine glitches in the raw logs. Catching them is useful
as a data-quality check before trusting the telemetry for anything else.

## Structure

```
src/
  load.py       # wide multi-drone CSV -> long-form time series
  features.py   # speed, vertical rate, battery rate, sample-gap
  detect.py     # rule-based + Isolation Forest, combined
  explain.py    # SHAP explanations of the Isolation Forest
  run.py        # full pipeline: load -> features -> detect -> report/plots
data/
  stage16_telemetry/   # telemetry CSVs go here
results/               # summaries, flagged samples, plots
requirements.txt
```

## Explainability

`explain.py` uses SHAP to show why the Isolation Forest scores a sample the way
it does — overall feature importance, a beeswarm, and a breakdown for one
sample. The findings, and a note on a sign-convention trap I ran into, are in
[`EXPLAINABILITY.md`](EXPLAINABILITY.md).

```bash
python -m src.explain --data data/stage16_telemetry --out results
```

## Notes

- Altitude rules are split into two: `alt_fault` for impossible values and
  `alt_envelope` for mild excursions, so real glitches don't get mixed in with
  small deviations.
- The Isolation Forest needs no labels, which fits telemetry where I don't have
  ground-truth fault labels.
- The rule thresholds are arguments in `detect.rule_flags`, so they're easy to
  change for a different drone's flight envelope.
