# Explainability (SHAP)

The detector tells me which samples are anomalies. This part looks at why,
using SHAP on the Isolation Forest, both overall and for a single sample.

Run it:

```bash
python -m src.explain --data data/stage16_telemetry --out results
```

## The sign-convention trap

Isolation Forest is unsupervised, so there's no class probability — only a
score. SHAP here explains scikit-learn's `decision_function`, and the catch is
that this function is higher for normal points and lower (negative) for
anomalies. So a raw SHAP value reads backwards from what you'd expect:

- positive → feature pushed the sample toward normal
- negative → feature pushed it toward anomalous

That's easy to misread, so the code prints both: the raw sklearn version and a
sign-flipped version where positive means "toward anomalous". I kept both
rather than silently flipping, so the direction is always clear.

## Overall

Mean absolute SHAP per feature (how much it moves the score, ignoring
direction):

| Feature | Mean abs SHAP |
| --- | --- |
| speed_xy (horizontal speed) | 0.74 |
| dt_gap (gap between samples) | 0.74 |
| batt (battery level) | 0.56 |
| speed_z (vertical speed) | 0.53 |
| batt_rate (battery change) | 0.32 |
| z (altitude) | 0.22 |

The beeswarm (`results/shap_beeswarm.png`) adds direction:

- speed_xy: high values push toward anomalous — fast horizontal motion is the
  main signal.
- batt: low values push toward anomalous — low battery gets flagged.
- dt_gap: a separate cluster pushes hard toward anomalous — these are the
  dropouts, where a sample came in far off the normal rate.

## One sample

For the most anomalous sample (`decision_function = -0.076`), every feature
pushed toward anomalous at once, led by speed_xy (+3.3 SD) and speed_z
(-5.7 SD), with altitude at -7.2 SD. So it's not one bad number — the sample is
extreme on several axes together. Plot: `results/shap_local_most_anomalous.png`.

## Rules vs. model

Worth noting: the rules flag altitude most, because of those few impossible
values. But SHAP shows the model leans more on speed and timing across the
whole dataset. They're looking at different things — the rules catch known hard
faults, the model catches broader patterns — which is why I run both instead of
picking one.

## Files

| File | What's in it |
| --- | --- |
| `shap_global_importance.csv` / `.png` | mean abs SHAP per feature |
| `shap_beeswarm.png` | every point, coloured by feature value |
| `shap_local_most_anomalous.csv` / `.png` | one sample, both sign conventions |

## Caveat

SHAP runs on a 3,000-sample random subset (fixed seed) for speed, not all
307k. The detection itself runs on the full set.
