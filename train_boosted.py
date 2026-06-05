from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


DATA_DIR = Path("dataset") if Path("dataset").exists() else Path("dataset/dataset")
OUTPUT_DIR = Path("outputs")


def decode_geohash(code: str) -> tuple[float, float]:
    alphabet = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    even = True
    for char in code:
        value = alphabet.index(char)
        for mask in [16, 8, 4, 2, 1]:
            interval = lon if even else lat
            midpoint = (interval[0] + interval[1]) / 2
            interval[0 if value & mask else 1] = midpoint
            even = not even
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


def prepare_features(
    fit_train: pd.DataFrame,
    frames: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[str]]:
    combined = pd.concat(frames, ignore_index=True).copy()
    combined["hour"] = combined["timestamp"].str.split(":").str[0].astype(int)
    combined["minute"] = combined["timestamp"].str.split(":").str[1].astype(int)
    combined["slot"] = combined["hour"] * 4 + combined["minute"] // 15
    combined["time_sin"] = np.sin(2 * np.pi * combined["slot"] / 96)
    combined["time_cos"] = np.cos(2 * np.pi * combined["slot"] / 96)
    combined["geo_prefix4"] = combined["geohash"].str[:4]
    combined["geo_prefix5"] = combined["geohash"].str[:5]

    coordinates = dict.fromkeys(combined["geohash"].unique())
    coordinates = {key: decode_geohash(key) for key in coordinates}
    combined["latitude"] = combined["geohash"].map(lambda value: coordinates[value][0])
    combined["longitude"] = combined["geohash"].map(lambda value: coordinates[value][1])

    reference = fit_train.copy()
    reference["hour"] = reference["timestamp"].str.split(":").str[0].astype(int)
    reference["minute"] = reference["timestamp"].str.split(":").str[1].astype(int)
    reference["slot"] = reference["hour"] * 4 + reference["minute"] // 15

    aggregations = [
        (["geohash"], ["mean", "std", "median", "min", "max"]),
        (["timestamp"], ["mean", "std", "median", "max"]),
        (["RoadType", "timestamp"], ["mean", "median", "std"]),
        (["NumberofLanes", "timestamp"], ["mean", "median"]),
        (["LargeVehicles", "timestamp"], ["mean", "median"]),
        (["Landmarks", "timestamp"], ["mean", "median"]),
        (["Weather", "timestamp"], ["mean", "median"]),
        (["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks"], ["mean", "median"]),
        (["geo_prefix4", "timestamp"], ["mean"]),
        (["geo_prefix5", "timestamp"], ["mean"]),
    ]
    reference["geo_prefix4"] = reference["geohash"].str[:4]
    reference["geo_prefix5"] = reference["geohash"].str[:5]
    for number, (keys, stats) in enumerate(aggregations):
        table = reference.groupby(keys, dropna=False)["demand"].agg(stats).reset_index()
        table.columns = keys + [f"agg_{number}_{stat}" for stat in stats]
        combined = combined.merge(table, on=keys, how="left")

    lag = reference[reference["day"] == 48][["geohash", "timestamp", "demand"]]
    lag = lag.rename(columns={"demand": "demand_day48"})
    combined = combined.merge(lag, on=["geohash", "timestamp"], how="left")

    categorical = [
        "geohash",
        "geo_prefix4",
        "geo_prefix5",
        "timestamp",
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
    ]
    for column in categorical:
        values = combined[column].fillna("Missing").astype(str)
        categories = {value: index for index, value in enumerate(sorted(values.unique()))}
        combined[column] = values.map(categories).astype(int)

    combined["Temperature"] = combined["Temperature"].fillna(reference["Temperature"].median())
    combined = combined.replace([np.inf, -np.inf], np.nan)
    numeric_columns = [
        column
        for column in combined.columns
        if column not in {"Index", "demand"} and pd.api.types.is_numeric_dtype(combined[column])
    ]
    combined[numeric_columns] = combined[numeric_columns].fillna(
        combined[numeric_columns].median(numeric_only=True)
    )

    lengths = [len(frame) for frame in frames]
    outputs = []
    start = 0
    for length in lengths:
        outputs.append(combined.iloc[start : start + length].reset_index(drop=True))
        start += length
    return outputs, numeric_columns


def build_model(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=1800,
        learning_rate=0.025,
        max_depth=8,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    day48 = train[train["day"] == 48].copy()
    day49 = train[train["day"] == 49].copy()
    prepared, features = prepare_features(day48, [day48, day49.drop(columns=["demand"])])
    x_train, x_valid = prepared
    validation_model = build_model()
    validation_model.fit(x_train[features], day48["demand"])
    validation_pred = np.clip(validation_model.predict(x_valid[features]), 0, 1)
    print(f"day48_to_day49_r2={r2_score(day49['demand'], validation_pred):.6f}")

    prepared, features = prepare_features(train, [train, test])
    x_train, x_test = prepared
    model = build_model()
    model.fit(x_train[features], train["demand"])
    prediction = np.clip(model.predict(x_test[features]), 0, 1)

    submission = test[["Index"]].copy()
    submission["demand"] = prediction
    submission.to_csv(OUTPUT_DIR / "submission_boosted.csv", index=False)
    print(submission.head().to_string(index=False))
    print(f"wrote {OUTPUT_DIR / 'submission_boosted.csv'}")


if __name__ == "__main__":
    main()
