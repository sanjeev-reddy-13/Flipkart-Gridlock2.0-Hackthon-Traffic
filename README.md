# Gridlock Hackathon 2.0 - Phase 1 Traffic Demand Prediction

## Problem Statement

This project predicts traffic demand for day 49 at 15-minute intervals from `02:15` to `13:45` for each geospatial cell in the test set. The training data includes all quarter-hour records for day 48 plus the first nine records of day 49, so the task is to use historical day-48 patterns and the earliest observed day-49 signal to forecast the remaining day-49 demand.

The output must be a submission file with the same `Index` values as `dataset/test.csv` and a predicted `demand` value between 0 and 1 for each row.

## Input Dataset

The repository expects data files in the `dataset/` folder.

- `dataset/train.csv` — 77,299 rows
  - `Index` (int64)
  - `geohash` (string)
  - `day` (int64)
  - `timestamp` (string)
  - `demand` (float64)
  - `RoadType` (string)
  - `NumberofLanes` (int64)
  - `LargeVehicles` (string)
  - `Landmarks` (string)
  - `Temperature` (float64)
  - `Weather` (string)

- `dataset/test.csv` — 41,778 rows
  - `Index` (int64)
  - `geohash` (string)
  - `day` (int64)
  - `timestamp` (string)
  - `RoadType` (string)
  - `NumberofLanes` (int64)
  - `LargeVehicles` (string)
  - `Landmarks` (string)
  - `Temperature` (float64)
  - `Weather` (string)

- `dataset/sample_submission.csv` — 5 rows
  - `Index` (int64)
  - `demand` (float64)

The `timestamp` values use a simple string format such as `0:0`, `2:15`, and `13:45`, and `day` indicates the calendar day number (48 or 49).

## Approach

The code uses two main modeling strategies:

1. `train_predict.py` — recursive temporal baseline
   - builds a day-48 demand surface indexed by `geohash` and `timestamp`
   - fills missing cells with linear interpolation within each geohash time series
   - for each day-49 prediction timestamp, applies a correction from the previous 15-minute interval:

```text
prediction = day48_current + beta * (previous_day49 - day48_previous)
final      = blend * previous_day49 + (1 - blend) * prediction
```

   - this recursive blend reduces drift while allowing the model to adapt to day-49 changes

2. `train_lgbm.py` — LightGBM ensemble
   - engineers features from geohash prefixes, cyclical time, day-48 lag demand, and aggregated demand statistics
   - trains a 3-seed LightGBM ensemble on the full training set
   - creates direct predictions and blended results with the temporal baseline

The repository also includes `train_boosted.py`, which trains an XGBoost model on engineered features.

## Validation

The validation is implemented in `train_predict.py` and is correctly described:

- it uses the known day-49 records before `02:15` as seed data
- it validates recursively from `00:15` to `02:00`, seeded by the observed `00:00` records
- it computes a strict recursive R² score for the seeded window
- it also computes validation scores for multiple future-window splits to measure robustness

The tuned recursive model parameters are:

- `beta = 0.85`
- `blend_previous = 0.55`

These values adjust the day-48 baseline using the most recent day-49 demand change while limiting prediction drift.

## Output

The scripts write generated submissions and diagnostics to the `outputs/` folder.

- `outputs/submission.csv` — primary recursive temporal submission
- `outputs/submission_balanced.csv` — balanced recursive variant
- `outputs/submission_conservative.csv` — smoother recursive variant
- `outputs/submission_persistent.csv` — higher persistence recursive variant
- `outputs/submission_boosted.csv` — XGBoost model submission
- `outputs/submission_lgbm.csv` — LightGBM ensemble submission
- `outputs/submission_lgbm_blend_*.csv` — blended LightGBM/temporal variants
- `outputs/diagnostics.json` — validation scores and diagnostic statistics
- `outputs/gridlock_phase1_source.zip` — packaged source archive from `train_predict.py`

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python train_predict.py
python train_lgbm.py
python train_boosted.py
```

If you only want the recursive temporal baseline, run `train_predict.py`. For the boosted models, run `train_lgbm.py` or `train_boosted.py`.

