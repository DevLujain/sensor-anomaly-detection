# Explainability (SHAP) — what drives the anomaly scores

The anomaly detector flags *which* telemetry samples are anomalous. This
write-up uses **SHAP (TreeSHAP)** on the Isolation Forest to explain *why*,
both globally (across the dataset) and locally (for a single sample).

Run it with:

```bash
python -m src.explain --data data/stage16_telemetry --out results
```

## A note on what is being explained (and the sign convention)

Isolation Forest is **unsupervised** — it has no class probability, only an
isolation-based score. SHAP here attributes scikit-learn's
`decision_function` output, where **higher = more normal** and
**lower (negative) = more anomalous**. That means a raw SHAP value reads as:

- positive → the feature pushed the sample **toward normal**
- negative → the feature pushed the sample **toward anomalous**

This is the reverse of the intuitive "what made it anomalous?" reading, and is
the most common way these plots get misinterpreted. The code therefore reports
**both** conventions side by side (`shap_sklearn (+=normal)` and
`shap_anomaly (+=anomalous)`) so the direction is never ambiguous.

## Global findings

Mean absolute SHAP per feature (impact on the score, direction-agnostic):

| Rank | Feature | Mean \|SHAP\| |
| --- | --- | --- |
| 1 | speed_xy (horizontal speed) | 0.74 |
| 2 | dt_gap (sampling interval) | 0.74 |
| 3 | batt (battery level) | 0.56 |
| 4 | speed_z (vertical speed) | 0.53 |
| 5 | batt_rate (battery change rate) | 0.32 |
| 6 | z (altitude) | 0.22 |

The **beeswarm plot** (`results/shap_beeswarm.png`) adds direction:

- **speed_xy**: high values (red) push strongly toward anomalous — fast
  horizontal motion is the dominant anomaly signal.
- **batt**: low values (blue) push toward anomalous — low battery is flagged,
  as expected.
- **dt_gap**: a distinct cluster of points pushes hard toward anomalous —
  the sensor-dropout signature (samples arriving far off the nominal rate).

## Local finding (most anomalous sample)

For the single most anomalous sample (`decision_function = -0.076`), **every
feature pushed toward anomalous at once** — led by speed_xy (+3.3 SD) and
speed_z (-5.7 SD), with altitude at -7.2 SD. It is anomalous not because of one
bad reading but because it is simultaneously extreme on several axes. See
`results/shap_local_most_anomalous.png`.

## An honest contrast worth noting

The rule-based detector (`detect.py`) flags **altitude** most often, because a
few physically impossible altitude values exist in the logs. But SHAP shows the
*model* leans more on **speed and timing** across the dataset as a whole. The
two views are complementary: rules catch known hard faults; the model captures
broader multivariate structure. Neither alone tells the full story — which is
the practical case for using explainability alongside detection rather than
trusting a single number.

## Files produced

| File | Contents |
| --- | --- |
| `shap_global_importance.csv` / `.png` | mean \|SHAP\| per feature |
| `shap_beeswarm.png` | per-point, value-coloured global view |
| `shap_local_most_anomalous.csv` / `.png` | single-sample attribution, both sign conventions |
