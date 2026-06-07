"""
run.py
End-to-end sensor anomaly-detection pipeline on Stage-16 drone telemetry.

    python -m src.run --data data/stage16_telemetry --out results

Outputs (in --out):
  - anomaly_summary.csv     per-rule and per-drone anomaly counts
  - anomalies.csv           every flagged sample with its reasons
  - timeline_<run>.png      battery/altitude with anomalies marked (sample run)
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .load import load_all
from .features import add_features
from .detect import detect, rule_flags

RULE_NAMES = ["alt_fault", "alt_envelope", "vspeed", "hspeed",
              "batt_jump", "sensor_dropout"]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    rows.append({"metric": "total_samples", "value": total})
    rows.append({"metric": "anomalies_total", "value": int(df["anomaly"].sum())})
    rows.append({"metric": "anomaly_rate_pct",
                 "value": round(100 * df["anomaly"].mean(), 3)})
    rows.append({"metric": "iso_anomalies", "value": int(df["iso_anomaly"].sum())})
    for r in RULE_NAMES:
        rows.append({"metric": f"rule_{r}", "value": int(df[r].sum())})
    return pd.DataFrame(rows)


def per_drone(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("drone")
              .agg(samples=("anomaly", "size"),
                   anomalies=("anomaly", "sum"),
                   anomaly_rate_pct=("anomaly", lambda s: round(100*s.mean(), 3)))
              .reset_index())


def plot_run(df: pd.DataFrame, run_id: str, drone: str, out_dir: str):
    sub = df[(df.run_id == run_id) & (df.drone == drone)].copy()
    if sub.empty:
        return None
    t = sub["ts"] - sub["ts"].min()
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax[0].plot(t, sub["batt"], lw=0.8, label="battery")
    ax[0].scatter(t[sub.anomaly], sub["batt"][sub.anomaly], s=10, c="red",
                  zorder=3, label="anomaly")
    ax[0].set_ylabel("battery (%)"); ax[0].legend(loc="upper right")
    ax[0].set_title(f"Stage-16 telemetry anomalies | run {run_id} | {drone}")

    ax[1].plot(t, sub["z"], lw=0.8, color="tab:blue", label="altitude z")
    ax[1].scatter(t[sub.anomaly], sub["z"][sub.anomaly], s=10, c="red", zorder=3)
    ax[1].set_ylabel("altitude (m)"); ax[1].set_xlabel("time since run start (s)")

    fig.tight_layout()
    path = os.path.join(out_dir, f"timeline_{run_id}_{drone}.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage16_telemetry")
    ap.add_argument("--out", default="results")
    ap.add_argument("--contamination", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading telemetry from {args.data} ...")
    long = load_all(args.data)
    print(f"  {len(long):,} samples | drones {sorted(long.drone.unique())} "
          f"| {long.run_id.nunique()} runs")

    print("Building features ...")
    feat = add_features(long)

    print("Detecting anomalies (rules + Isolation Forest) ...")
    res = detect(feat, contamination=args.contamination)

    summary = summarise(res)
    drone_tbl = per_drone(res)
    summary.to_csv(os.path.join(args.out, "anomaly_summary.csv"), index=False)
    drone_tbl.to_csv(os.path.join(args.out, "per_drone_summary.csv"), index=False)

    # export the flagged samples with their reasons
    reason_cols = ["ts", "day", "run_id", "drone", "state", "x", "y", "z",
                   "batt", *RULE_NAMES, "iso_anomaly", "iso_score"]
    anomalies = res[res["anomaly"]][reason_cols].copy()
    anomalies.to_csv(os.path.join(args.out, "anomalies.csv"), index=False)

    # plot the largest run as an illustrative timeline
    biggest = res.run_id.value_counts().idxmax()
    for d in sorted(res[res.run_id == biggest].drone.unique()):
        p = plot_run(res, biggest, d, args.out)
        if p:
            print("  wrote", p)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PER DRONE ===")
    print(drone_tbl.to_string(index=False))
    print(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
