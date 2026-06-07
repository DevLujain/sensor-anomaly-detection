"""
explain.py
SHAP explanations for the Isolation Forest anomaly detector.

WHAT THIS DOES, AND THE REASONING BEHIND EACH CHOICE
----------------------------------------------------
The anomaly detector (detect.py) tells us WHICH samples are anomalous.
It does not tell us WHY. This module answers "why" with SHAP.

Key design decisions (made deliberately, not by default):

1. WHAT we explain:
   We explain the Isolation Forest's `decision_function` output using
   TreeSHAP. Note: this is unusual because Isolation Forest is UNSUPERVISED
   - there is no class probability, only an isolation-based score.

2. THE SIGN CONVENTION (the important subtlety):
   In scikit-learn, `decision_function` is HIGHER for normal points and
   LOWER (negative) for anomalies. So a SHAP value on this output reads as:
       positive SHAP  -> feature pushed the point toward NORMAL
       negative SHAP  -> feature pushed the point toward ANOMALOUS
   This is the opposite of the intuitive "which features made it anomalous?"
   reading. Rather than silently flipping it, we show BOTH conventions and
   label them, so the interpretation is explicit and not mis-read.

3. HOW MUCH data:
   TreeSHAP is exact for trees but still scales with samples x trees. We
   compute global importance on a random sample (default 3000) for speed,
   and local explanations on individually chosen rows. The sampling seed is
   fixed for reproducibility.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .load import load_all
from .features import add_features, feature_matrix, FEATURE_COLS


# ---------------------------------------------------------------------------
def fit_model(folder: str, contamination: float = 0.01, seed: int = 42):
    """Reproduce the anomaly detector exactly, returning model + scaled data."""
    df = add_features(load_all(folder))
    X = feature_matrix(df)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    iso = IsolationForest(contamination=contamination,
                          random_state=seed, n_estimators=200).fit(Xs)
    return df, Xs, iso


def shap_values_for(iso, Xs, sample_size=3000, seed=42):
    """
    Compute TreeSHAP values on a random sample of the data.

    Returns (shap_raw, idx) where shap_raw explains decision_function:
      positive -> toward NORMAL, negative -> toward ANOMALOUS  (sklearn sign)
    """
    import shap
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(Xs))
    idx = rng.choice(len(Xs), size=n, replace=False)
    explainer = shap.TreeExplainer(iso)
    shap_raw = explainer.shap_values(Xs[idx])
    return np.asarray(shap_raw), idx


# ---------------------------------------------------------------------------
# GLOBAL explanation
# ---------------------------------------------------------------------------
def global_importance(shap_raw) -> pd.DataFrame:
    """
    Mean absolute SHAP per feature = overall importance.
    |SHAP| is sign-agnostic, so this is the same under both conventions -
    it answers "which features matter most", not "in which direction".
    """
    imp = np.abs(shap_raw).mean(axis=0)
    return (pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": imp})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))


def plot_global(importance: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1],
            color="tab:blue")
    ax.set_xlabel("mean |SHAP| (impact on anomaly score)")
    ax.set_title("Global feature importance \u2014 Isolation Forest (TreeSHAP)")
    fig.tight_layout()
    p = os.path.join(out_dir, "shap_global_importance.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# LOCAL explanation (single sample), shown in BOTH sign conventions
# ---------------------------------------------------------------------------
def local_explanation(shap_raw, idx, Xs, iso, pick="most_anomalous"):
    """
    Explain one sample. Returns a tidy DataFrame with both conventions.

      shap_sklearn : raw decision_function attribution
                     (+ = toward normal, - = toward anomalous)
      shap_anomaly : sign-flipped so (+ = toward anomalous) - the intuitive read
    """
    dec = iso.decision_function(Xs[idx])           # aligned to the sampled rows
    if pick == "most_anomalous":
        local_pos = int(np.argmin(dec))            # lowest decision_function
    elif pick == "most_normal":
        local_pos = int(np.argmax(dec))
    else:
        local_pos = int(pick)

    row_shap = shap_raw[local_pos]
    out = pd.DataFrame({
        "feature": FEATURE_COLS,
        "value_scaled": Xs[idx][local_pos],
        "shap_sklearn (+=normal)": row_shap,
        "shap_anomaly (+=anomalous)": -row_shap,
    })
    out = out.reindex(out["shap_anomaly (+=anomalous)"].abs()
                      .sort_values(ascending=False).index).reset_index(drop=True)
    return out, local_pos, float(dec[local_pos])


def plot_beeswarm(shap_raw, idx, Xs, out_dir: str):
    """
    SHAP beeswarm: every sampled point, every feature, coloured by feature
    value. Shows not just WHICH features matter but HOW a feature's value
    relates to the score. We plot in the anomaly-oriented convention
    (sign-flipped: right = pushes toward anomalous).
    """
    import shap
    shap_anom = -shap_raw  # flip to "+ = toward anomalous"
    fig = plt.figure()
    shap.summary_plot(
        shap_anom, features=Xs[idx], feature_names=FEATURE_COLS,
        show=False, plot_size=(8, 4.5),
    )
    plt.title("SHAP beeswarm \u2014 toward anomalous (right)")
    plt.tight_layout()
    p = os.path.join(out_dir, "shap_beeswarm.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    return p


def plot_local(local_df: pd.DataFrame, out_dir: str):
    """Plot the intuitive (anomaly-oriented) attribution for one sample."""
    d = local_df.iloc[::-1]
    colors = ["tab:red" if v > 0 else "tab:green"
              for v in d["shap_anomaly (+=anomalous)"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d["feature"], d["shap_anomaly (+=anomalous)"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP toward anomalous (red) vs. normal (green)")
    ax.set_title("Local explanation \u2014 most anomalous sample")
    fig.tight_layout()
    p = os.path.join(out_dir, "shap_local_most_anomalous.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage16_telemetry")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Fitting the anomaly detector (same setup as detect.py) ...")
    df, Xs, iso = fit_model(args.data)

    print(f"Computing TreeSHAP on a {args.sample}-sample subset ...")
    shap_raw, idx = shap_values_for(iso, Xs, sample_size=args.sample)

    # GLOBAL
    imp = global_importance(shap_raw)
    imp.to_csv(os.path.join(args.out, "shap_global_importance.csv"), index=False)
    p1 = plot_global(imp, args.out)
    print("\n=== GLOBAL IMPORTANCE (mean |SHAP|) ===")
    print(imp.to_string(index=False))
    print("wrote", p1)

    # BEESWARM (global, per-point, value-coloured)
    pbee = plot_beeswarm(shap_raw, idx, Xs, args.out)
    print("wrote", pbee)

    # LOCAL (most anomalous sample), both conventions
    local_df, pos, dec_val = local_explanation(shap_raw, idx, Xs, iso,
                                               pick="most_anomalous")
    local_df.to_csv(os.path.join(args.out, "shap_local_most_anomalous.csv"),
                    index=False)
    p2 = plot_local(local_df, args.out)
    print(f"\n=== LOCAL EXPLANATION (most anomalous; decision_function={dec_val:.3f}) ===")
    print(local_df.to_string(index=False))
    print("wrote", p2)

    print(f"\nResults saved to {args.out}/")
    print("\nReading guide:")
    print("  shap_sklearn (+=normal)   : raw sklearn convention")
    print("  shap_anomaly (+=anomalous): sign-flipped, intuitive read")


if __name__ == "__main__":
    main()
