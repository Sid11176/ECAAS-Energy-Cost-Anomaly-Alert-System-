"""
ECAAS Stage 1 — Forecast Service
Trains a demand forecast model on the cleaned Cebu hourly load data.

Model: XGBoost regressor predicting demand_mw for a given timestamp.
Confidence band: residual standard deviation bucketed by hour-of-day,
                  used to build a ~90% interval around the point forecast.

Output: models/forecast_model.json (booster) + models/residual_bands.json
"""
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA_PATH = "data/cebu_hourly_demand_2013_2025_long.csv"
MODEL_PATH = "models/forecast_model.json"
BANDS_PATH = "models/residual_bands.json"
FEATURE_COLS_PATH = "models/feature_columns.json"


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek          # 0=Mon
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dayofyear"] = ts.dt.dayofyear
    df["year"] = ts.dt.year

    # cyclical encodings so the model sees hour 23 and hour 0 as close
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # lag features (all lags are safe: computed from strictly past readings)
    df["lag_1h"] = df["demand_mw"].shift(1)
    df["lag_24h"] = df["demand_mw"].shift(24)          # same hour, previous day
    df["lag_168h"] = df["demand_mw"].shift(168)         # same hour, previous week
    df["roll_mean_24h"] = df["demand_mw"].shift(1).rolling(24).mean()
    df["roll_mean_168h"] = df["demand_mw"].shift(1).rolling(168).mean()

    # only drop rows where a FEATURE or the target is missing (lag warm-up
    # window at the very start). spot_peak_mw is daily-only and unused here,
    # so it must not trigger drops.
    required_cols = FEATURE_COLS + ["demand_mw"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


FEATURE_COLS = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "dayofweek", "is_weekend", "month", "year",
    "lag_1h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_mean_168h",
]


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} raw rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print("Engineering features...")
    feat = engineer_features(df)
    print(f"  {len(feat)} rows after lag warm-up window dropped")

    X = feat[FEATURE_COLS]
    y = feat["demand_mw"]

    # time-based split: last 6 months as holdout, never shuffle time series
    split_date = feat["timestamp"].max() - pd.Timedelta(days=180)
    train_mask = feat["timestamp"] < split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    print(f"  train: {len(X_train)} rows, test: {len(X_test)} rows (holdout = last 180 days)")

    print("Training XGBoost regressor...")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    print(f"  Holdout MAE:  {mae:.2f} MW")
    print(f"  Holdout MAPE: {mape*100:.2f}%")

    # ---- confidence band calibration ----
    # residuals bucketed by hour-of-day capture the fact that forecast error
    # is not constant across the day (e.g. ramp hours are noisier than 3am)
    resid = y_test.values - preds
    test_hours = feat.loc[~train_mask, "hour"].values
    bands = {}
    for h in range(24):
        h_resid = resid[test_hours == h]
        if len(h_resid) > 5:
            bands[str(h)] = {
                "std": float(np.std(h_resid)),
                "n": int(len(h_resid)),
            }
        else:
            bands[str(h)] = {"std": float(np.std(resid)), "n": int(len(resid))}

    print("Saving model + calibration artifacts...")
    model.save_model(MODEL_PATH)
    with open(BANDS_PATH, "w") as f:
        json.dump(bands, f, indent=2)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print("Done.")
    print(f"  model:   {MODEL_PATH}")
    print(f"  bands:   {BANDS_PATH}")
    print(f"  columns: {FEATURE_COLS_PATH}")


if __name__ == "__main__":
    main()
