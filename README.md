# Gridlock Hackathon 2.0 - Phase 1 Traffic Demand Prediction

## Approach

The train/test split is temporal:

- `train.csv` contains all quarter-hour records for day 48 and the first nine records of day 49.
- `test.csv` asks for day 49 demand from `02:15` to `13:45`.

The solution builds a day-48 geohash-by-time demand surface and uses it as the base traffic pattern. For each test timestamp, it applies a recursive day-49 correction from the previous 15-minute interval:

```text
prediction = day48_current + beta * (previous_day49 - day48_previous)
final      = blend * previous_day49 + (1 - blend) * prediction
```

This captures the observed demand shift from day 48 to day 49 while reducing drift during recursive prediction.

## Validation

The local validation predicts the known day-49 window from `00:15` to `02:00`, seeded by the observed `00:00` records. The tuned parameters are:

- `beta = 0.90`
- `blend_previous = 0.50`

Local recursive validation R2 is printed when running `train_predict.py`.

## Files

- `train_predict.py`: full training, validation, prediction, and packaging script.
- `outputs/submission.csv`: prediction file for Hackerearth upload.
- `outputs/gridlock_phase1_source.zip`: source archive for the source-code upload section.

## Run

```bash
python train_predict.py
```
