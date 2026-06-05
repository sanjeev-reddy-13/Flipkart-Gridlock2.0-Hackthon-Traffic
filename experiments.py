from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


DATA = Path("dataset/dataset")


def time_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def decode_geohash(code: str) -> tuple[float, float]:
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    even = True
    for char in code:
        bits = base32.index(char)
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lon[0] + lon[1]) / 2
                if bits & mask:
                    lon[0] = mid
                else:
                    lon[1] = mid
            else:
                mid = (lat[0] + lat[1]) / 2
                if bits & mask:
                    lat[0] = mid
                else:
                    lat[1] = mid
            even = not even
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


def add_features(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ref = train.copy()
    ref["minute"] = ref["timestamp"].map(time_minutes)
    ref["hour"] = ref["minute"] // 60

    out["minute"] = out["timestamp"].map(time_minutes)
    out["hour"] = out["minute"] // 60
    out["quarter"] = (out["minute"] % 60) // 15
    out["sin_time"] = np.sin(2 * np.pi * out["minute"] / 1440)
    out["cos_time"] = np.cos(2 * np.pi * out["minute"] / 1440)

    geo = pd.DataFrame(
        [decode_geohash(g) for g in out["geohash"]],
        columns=["lat", "lon"],
        index=out.index,
    )
    out = pd.concat([out, geo], axis=1)

    day48 = train[train["day"] == 48][["geohash", "timestamp", "demand"]]
    out = out.merge(
        day48.rename(columns={"demand": "lag_day48_same_time"}),
        on=["geohash", "timestamp"],
        how="left",
    )
    out = out.merge(
        train.groupby("geohash")["demand"]
        .agg(["mean", "median", "std", "max"])
        .rename(
            columns={
                "mean": "geo_mean",
                "median": "geo_median",
                "std": "geo_std",
                "max": "geo_max",
            }
        ),
        on="geohash",
        how="left",
    )
    out = out.merge(
        train.groupby("timestamp")["demand"]
        .mean()
        .rename("time_mean"),
        on="timestamp",
        how="left",
    )
    out = out.merge(
        ref.groupby(["geohash", "hour"])["demand"]
        .mean()
        .rename("geo_hour_mean"),
        on=["geohash", "hour"],
        how="left",
    )
    out = out.merge(
        train.groupby(["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Weather"])[
            "demand"
        ]
        .mean()
        .rename("road_weather_mean"),
        on=["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Weather"],
        how="left",
    )

    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather", "geohash"]:
        out[col] = out[col].fillna("missing").astype("category").cat.codes
    return out


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    holdout = train[train["day"] == 49].copy()
    base = add_features(holdout.drop(columns=["demand"]), train[train["day"] == 48])
    lag_pred = base["lag_day48_same_time"].fillna(base["geo_mean"]).fillna(train["demand"].mean())
    print("lag baseline r2", r2_score(holdout["demand"], lag_pred))

    dev = add_features(train[train["day"] == 48].drop(columns=["demand"]), train[train["day"] == 48])
    y = train[train["day"] == 48]["demand"]
    val = add_features(holdout.drop(columns=["demand"]), train[train["day"] == 48])
    features = [
        c
        for c in dev.columns
        if c not in ["Index", "timestamp"] and dev[c].dtype.kind in "biufc"
    ]
    model = XGBRegressor(
        n_estimators=900,
        max_depth=6,
        learning_rate=0.025,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=4,
        reg_lambda=3.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(dev[features], y)
    pred = np.clip(model.predict(val[features]), 0, 1)
    print("xgb day48->day49 early r2", r2_score(holdout["demand"], pred))
    for w in np.linspace(0, 1, 11):
        blend = w * pred + (1 - w) * lag_pred
        print(f"blend xgb={w:.1f}", r2_score(holdout["demand"], np.clip(blend, 0, 1)))


if __name__ == "__main__":
    main()
