"""
load.py
Load Stage-16 drone telemetry CSVs (wide, multi-drone) and reshape into a
tidy long-form time series: one row per (timestamp, drone).

Raw wide schema (per file):
    ts, day, D0_state, D0_x, D0_y, D0_z, D0_batt, D1_state, D1_x, ...

Long schema produced here:
    ts, day, drone, state, x, y, z, batt, run_id, dt
"""

from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd

DRONE_RE = re.compile(r"^D(\d+)_(.+)$")


def _detect_drones(columns) -> list[str]:
    return sorted({m.group(1) for c in columns if (m := DRONE_RE.match(c))},
                  key=int)


def load_one(path: str) -> pd.DataFrame:
    """Load a single wide telemetry CSV into long form."""
    df = pd.read_csv(path)
    run_id = os.path.basename(path).replace("telemetry_", "").replace(".csv", "")
    drones = _detect_drones(df.columns)

    frames = []
    for d in drones:
        cols = {
            f"D{d}_state": "state",
            f"D{d}_x": "x",
            f"D{d}_y": "y",
            f"D{d}_z": "z",
            f"D{d}_batt": "batt",
        }
        present = {k: v for k, v in cols.items() if k in df.columns}
        sub = df[["ts", "day", *present.keys()]].rename(columns=present).copy()
        sub["drone"] = f"D{d}"
        sub["run_id"] = run_id
        frames.append(sub)

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["drone", "ts"]).reset_index(drop=True)
    # per-drone sample interval
    long["dt"] = long.groupby("drone")["ts"].diff()
    return long


def load_all(folder: str, pattern: str = "telemetry_*.csv") -> pd.DataFrame:
    """Load and concatenate every telemetry file in a folder."""
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No telemetry files matching {pattern!r} in {folder!r}. "
            f"Put your Stage-16 telemetry CSVs there."
        )
    frames = [load_one(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    # numeric coercion (battery/positions occasionally blank during transitions)
    for c in ["x", "y", "z", "batt"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/stage16_telemetry"
    df = load_all(folder)
    print(f"loaded {len(df):,} rows  | drones: {sorted(df.drone.unique())}  "
          f"| runs: {df.run_id.nunique()}")
    print(df.head())
