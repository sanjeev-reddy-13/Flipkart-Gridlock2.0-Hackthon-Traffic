from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


DATA_DIR = Path("dataset/dataset")
OUTPUT_DIR = Path("outputs")
BETA = 0.90
BLEND_PREVIOUS = 0.50


def time_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def time_label(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60}"


def build_day48_surface(train: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    day48 = train[train["day"] == 48].copy()
    pivot = day48.pivot_table(
        index="geohash",
        columns="timestamp",
        values="demand",
        aggfunc="mean",
    )
    global_mean = float(day48["demand"].mean())
    geo_mean = pivot.mean(axis=1)
    time_mean = pivot.mean(axis=0)
    additive = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for timestamp in pivot.columns:
        additive[timestamp] = geo_mean + time_mean[timestamp] - global_mean
    surface = pivot.combine_first(additive).clip(0, 1)
    return surface, global_mean


def surface_lookup(surface: pd.DataFrame, geohash: str, timestamp: str, fallback: float) -> float:
    try:
        value = surface.loc[geohash, timestamp]
    except KeyError:
        return fallback
    if pd.isna(value):
        return fallback
    return float(value)


def recursive_predict(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    surface: pd.DataFrame,
    fallback: float,
) -> pd.Series:
    rows = rows.copy()
    rows["minute"] = rows["timestamp"].map(time_minutes)

    history = {
        (r.geohash, time_minutes(r.timestamp)): float(r.demand)
        for r in train[train["day"] == 49].itertuples(index=False)
    }
    predictions: dict[int, float] = {}

    for minute in sorted(rows["minute"].unique()):
        current = rows[rows["minute"] == minute]
        prev_timestamp = time_label(minute - 15)
        for row in current.itertuples(index=False):
            base_now = surface_lookup(surface, row.geohash, row.timestamp, fallback)
            base_prev = surface_lookup(surface, row.geohash, prev_timestamp, fallback)
            prev49 = history.get((row.geohash, minute - 15))

            if prev49 is None:
                pred = base_now
            else:
                corrected = base_now + BETA * (prev49 - base_prev)
                pred = (1 - BLEND_PREVIOUS) * corrected + BLEND_PREVIOUS * prev49

            pred = float(np.clip(pred, 0, 1))
            predictions[row.Index] = pred
            history[(row.geohash, minute)] = pred

    return pd.Series(predictions, name="demand")


def validate_recursive(train: pd.DataFrame, surface: pd.DataFrame, fallback: float) -> float:
    day49 = train[train["day"] == 49].copy()
    day49["minute"] = day49["timestamp"].map(time_minutes)
    validation_rows = day49[day49["minute"] > 0].drop(columns=["demand"])
    seed = pd.concat(
        [
            train[train["day"] == 48],
            day49[day49["minute"] == 0].drop(columns=["minute"]),
        ],
        ignore_index=True,
    )
    pred = recursive_predict(validation_rows, seed, surface, fallback)
    actual = day49[day49["minute"] > 0].set_index("Index")["demand"].loc[pred.index]
    return float(r2_score(actual, pred))


def write_source_zip() -> None:
    files = ["train_predict.py", "README.md"]
    with ZipFile(OUTPUT_DIR / "gridlock_phase1_source.zip", "w", ZIP_DEFLATED) as archive:
        for file_name in files:
            archive.write(file_name)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    surface, fallback = build_day48_surface(train)
    validation_r2 = validate_recursive(train, surface, fallback)
    print(f"recursive_validation_r2={validation_r2:.6f}")

    pred = recursive_predict(test, train, surface, fallback)
    submission = test[["Index"]].copy()
    submission["demand"] = submission["Index"].map(pred).astype(float).clip(0, 1)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    diagnostics = {
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "missing_predictions": int(submission["demand"].isna().sum()),
        "prediction_min": float(submission["demand"].min()),
        "prediction_mean": float(submission["demand"].mean()),
        "prediction_max": float(submission["demand"].max()),
        "recursive_validation_r2": validation_r2,
        "beta": BETA,
        "blend_previous": BLEND_PREVIOUS,
    }
    pd.Series(diagnostics).to_json(OUTPUT_DIR / "diagnostics.json", indent=2)
    write_source_zip()
    print(submission.head().to_string(index=False))
    print(f"wrote {OUTPUT_DIR / 'submission.csv'}")
    print(f"wrote {OUTPUT_DIR / 'gridlock_phase1_source.zip'}")


if __name__ == "__main__":
    main()
