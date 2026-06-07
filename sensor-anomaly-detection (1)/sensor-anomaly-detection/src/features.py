"""
features.py
Turn tidy telemetry into per-sample features for anomaly detection.

Derived signals (all per-drone, time-ordered):
  - speed_xy, speed_z : finite-difference velocities from position
  - batt_rate         : battery change per second (charging vs draining)
  - dt_gap            : sample interval (detects sensor dropouts)
  - z, batt           : raw altitude and battery level
These feed both a rule-based detector and an unsupervised model.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLS = ["speed_xy", "speed_z", "batt_rate", "dt_gap", "z", "batt"]


def add_features(long: pd.DataFrame) -> pd.DataFrame:
    df = long.sort_values(["run_id", "drone", "ts"]).copy()
    g = df.groupby(["run_id", "drone"], sort=False)

    dx = g["x"].diff()
    dy = g["y"].diff()
    dz = g["z"].diff()
    dt = g["ts"].diff()
    dt_safe = dt.replace(0, np.nan)

    df["dt_gap"] = dt
    df["speed_xy"] = np.sqrt(dx ** 2 + dy ** 2) / dt_safe
    df["speed_z"] = dz / dt_safe
    df["batt_rate"] = g["batt"].diff() / dt_safe

    # first sample of each group has no diff -> fill with neutral values
    df[["speed_xy", "speed_z", "batt_rate"]] = (
        df[["speed_xy", "speed_z", "batt_rate"]].fillna(0.0)
    )
    df["dt_gap"] = df["dt_gap"].fillna(df["dt_gap"].median())
    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    # guard against any residual NaN/inf from blank rows
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X
