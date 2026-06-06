"""
detect.py
Two complementary anomaly detectors for drone telemetry:

1. Rule-based (interpretable, physics-grounded):
   - battery dropout / impossible jump
   - altitude out of operating envelope or impossible vertical speed
   - sensor dropout (sample gap far larger than nominal)
   - implausible horizontal speed
   These encode domain knowledge an operator would trust.

2. Isolation Forest (unsupervised, learned):
   - flags multivariate outliers across all features, catching combinations
     no single rule describes.

The two are combined into a final per-sample anomaly label + score.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .features import FEATURE_COLS, feature_matrix


# ---------------------------------------------------------------------------
# 1. Rule-based detector
# ---------------------------------------------------------------------------
def rule_flags(df: pd.DataFrame,
               z_operating: float = 4.5,
               z_impossible: float = 6.0,
               z_floor: float = -0.1,
               max_speed_xy: float = 5.0,
               max_speed_z: float = 3.0,
               max_batt_jump: float = 30.0,
               nominal_dt: float = 0.5,
               dropout_factor: float = 4.0) -> pd.DataFrame:
    """Return a DataFrame of boolean rule flags aligned to df.

    Altitude is split into two severities:
      - alt_fault    : physically impossible reading (hard sensor glitch),
                       e.g. tens or thousands of metres, or large negative.
      - alt_envelope : mild excursion just outside the normal operating band.
    """
    flags = pd.DataFrame(index=df.index)
    flags["alt_fault"] = (df["z"] > z_impossible) | (df["z"] < z_floor - 1.0)
    flags["alt_envelope"] = (
        ((df["z"] > z_operating) & (df["z"] <= z_impossible))
        | ((df["z"] < z_floor) & (df["z"] >= z_floor - 1.0))
    )
    flags["vspeed"] = df["speed_z"].abs() > max_speed_z
    flags["hspeed"] = df["speed_xy"] > max_speed_xy
    # battery should change smoothly; large per-step jump = glitch
    flags["batt_jump"] = (df["batt_rate"].abs() * df["dt_gap"]) > max_batt_jump
    # sensor dropout: gap much larger than nominal sampling interval
    flags["sensor_dropout"] = df["dt_gap"] > (dropout_factor * nominal_dt)
    flags["rule_any"] = flags.any(axis=1)
    return flags


# ---------------------------------------------------------------------------
# 2. Isolation Forest detector
# ---------------------------------------------------------------------------
def isolation_forest_scores(df: pd.DataFrame,
                            contamination: float = 0.01,
                            random_state: int = 42):
    """Return (anomaly_bool, score) from an Isolation Forest over features."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = feature_matrix(df)
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination=contamination,
                          random_state=random_state, n_estimators=200)
    pred = iso.fit_predict(Xs)              # -1 anomaly, 1 normal
    score = -iso.score_samples(Xs)          # higher = more anomalous
    return (pred == -1), score


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def detect(df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    flags = rule_flags(out)
    iso_bool, iso_score = isolation_forest_scores(out, contamination)

    out = pd.concat([out, flags], axis=1)
    out["iso_anomaly"] = iso_bool
    out["iso_score"] = iso_score
    # final: flagged if either the rules or the model fire
    out["anomaly"] = out["rule_any"] | out["iso_anomaly"]
    return out
