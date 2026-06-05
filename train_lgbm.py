from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from train_boosted import prepare_features


DATA_DIR = Path("dataset") if Path("dataset").exists() else Path("dataset/dataset")
OUTPUT_DIR = Path("outputs")


def build_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3500,
        learning_rate=0.018,
        num_leaves=96,
        max_depth=9,
        min_child_samples=35,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.78,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    prepared, features = prepare_features(train, [train, test])
    x_train, x_test = prepared

    train_idx, valid_idx = train_test_split(
        np.arange(len(train)),
        test_size=0.18,
        random_state=42,
    )
    validation_model = build_model(42)
    validation_model.fit(
        x_train.iloc[train_idx][features],
        np.log1p(train.iloc[train_idx]["demand"]),
        eval_set=[(
            x_train.iloc[valid_idx][features],
            np.log1p(train.iloc[valid_idx]["demand"]),
        )],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    validation_prediction = np.expm1(
        validation_model.predict(
            x_train.iloc[valid_idx][features],
            num_iteration=validation_model.best_iteration_,
        )
    )
    print(f"random_holdout_r2={r2_score(train.iloc[valid_idx]['demand'], validation_prediction):.6f}")

    predictions = []
    for seed in [17, 42, 91]:
        model = build_model(seed)
        model.fit(x_train[features], np.log1p(train["demand"]))
        predictions.append(np.expm1(model.predict(x_test[features])))
    direct = np.clip(np.mean(predictions, axis=0), 0, 1)

    submission = test[["Index"]].copy()
    submission["demand"] = direct
    submission.to_csv(OUTPUT_DIR / "submission_lgbm.csv", index=False)

    temporal = pd.read_csv(OUTPUT_DIR / "submission.csv")["demand"].to_numpy()
    for direct_weight in [0.60, 0.75, 0.90]:
        blend = np.clip(direct_weight * direct + (1 - direct_weight) * temporal, 0, 1)
        blend_submission = test[["Index"]].copy()
        blend_submission["demand"] = blend
        suffix = int(direct_weight * 100)
        blend_submission.to_csv(OUTPUT_DIR / f"submission_lgbm_blend_{suffix}.csv", index=False)

    print(submission.head().to_string(index=False))
    print(f"wrote {OUTPUT_DIR / 'submission_lgbm.csv'}")


if __name__ == "__main__":
    main()
