from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


DATA = Path("dataset/dataset")


def time_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def time_label(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60}"


def make_surface(train: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    day48 = train[train.day == 48].copy()
    pivot = day48.pivot_table(index="geohash", columns="timestamp", values="demand", aggfunc="mean")
    global_mean = day48["demand"].mean()
    geo_mean = pivot.mean(axis=1)
    time_mean = pivot.mean(axis=0)
    additive = geo_mean.to_frame("_geo").join(time_mean.to_frame().T, how="outer")
    additive = additive.drop(columns=["_geo"]).add(geo_mean - global_mean, axis=0)
    additive = additive.fillna(global_mean)
    return pivot.combine_first(additive), global_mean


def lookup(surface: pd.DataFrame, geohash: str, timestamp: str, fallback: float) -> float:
    try:
        value = surface.loc[geohash, timestamp]
    except KeyError:
        return fallback
    if pd.isna(value):
        return fallback
    return float(value)


def predict_known_window(
    train: pd.DataFrame,
    surface: pd.DataFrame,
    fallback: float,
    beta: float,
    blend_prev: float,
    use_pred_history: bool,
) -> float:
    known = train[train.day == 49].copy()
    known["minute"] = known.timestamp.map(time_minutes)
    actual = {(r.geohash, r.minute): float(r.demand) for r in known.itertuples(index=False)}
    history = dict(actual)
    base = {
        (g, time_minutes(t)): float(v)
        for g, row in surface.iterrows()
        for t, v in row.items()
        if not pd.isna(v)
    }
    scores = []
    predictions = []
    for minute in sorted(known.minute.unique()):
        if minute == 0:
            continue
        current = known[known.minute == minute].copy()
        rows = []
        for row in current.itertuples(index=False):
            base_now = base.get((row.geohash, minute), fallback)
            base_prev = base.get((row.geohash, minute - 15), fallback)
            prev49 = history.get((row.geohash, minute - 15)) if use_pred_history else actual.get((row.geohash, minute - 15))
            if prev49 is not None:
                pred = base_now + beta * (prev49 - base_prev)
                pred = (1 - blend_prev) * pred + blend_prev * prev49
            else:
                pred = base_now
            rows.append(np.clip(pred, 0, 1))
        score = r2_score(current.demand, rows)
        scores.append(score)
        predictions.extend(zip(current.demand, rows))
        if use_pred_history:
            for row, pred in zip(current.itertuples(index=False), rows):
                history[(row.geohash, minute)] = float(pred)
    y, p = zip(*predictions)
    return r2_score(y, p), float(np.mean(scores))


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    surface, fallback = make_surface(train)
    best = None
    for beta in np.linspace(0.0, 1.5, 16):
        for blend_prev in np.linspace(0.0, 0.5, 6):
            overall, mean_slot = predict_known_window(train, surface, fallback, beta, blend_prev, use_pred_history=True)
            item = (overall, mean_slot, beta, blend_prev)
            if best is None or item > best:
                best = item
            print(f"beta={beta:.2f} blend_prev={blend_prev:.2f} overall={overall:.5f} slot_mean={mean_slot:.5f}")
    print("BEST", best)


if __name__ == "__main__":
    main()
