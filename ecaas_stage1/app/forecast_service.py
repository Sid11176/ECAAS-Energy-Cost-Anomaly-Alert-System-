"""
Core forecast logic, decoupled from FastAPI so it can be unit tested and
reused by the Stage 2 deviation engine directly.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from db import get_connection

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_VERSION = "xgb_v1_2013_2025"

_model = None
_bands = None
_feature_cols = None


def _load_artifacts():
    global _model, _bands, _feature_cols
    if _model is None:
        _model = XGBRegressor()
        _model.load_model(MODELS_DIR / "forecast_model.json")
        with open(MODELS_DIR / "residual_bands.json") as f:
            _bands = json.load(f)
        with open(MODELS_DIR / "feature_columns.json") as f:
            _feature_cols = json.load(f)
    return _model, _bands, _feature_cols


class InsufficientHistoryError(Exception):
    """Raised when there isn't 168h of prior readings to build lag features."""
    pass


def _fetch_history(conn, grid: str, before_ts: datetime, hours: int = 168):
    """Most recent `hours` readings strictly before before_ts, oldest first."""
    cur = conn.execute(
        """
        SELECT timestamp, demand_mw FROM readings
        WHERE grid = ? AND timestamp < ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (grid, before_ts.strftime("%Y-%m-%d %H:%M:%S"), hours),
    )
    rows = cur.fetchall()
    rows = list(reversed(rows))  # oldest first
    return rows


def build_features(target_ts: datetime, grid: str = "Cebu", conn=None):
    """
    Build the feature row for target_ts using the most recent readings
    available before it. Falls back to DB connection if none supplied.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    try:
        history = _fetch_history(conn, grid, target_ts, hours=168)
        if len(history) < 168:
            raise InsufficientHistoryError(
                f"Need 168h of prior readings for {grid} before {target_ts}, "
                f"only found {len(history)}."
            )

        values = [r["demand_mw"] for r in history]  # oldest -> newest, len 168
        lag_1h = values[-1]
        lag_24h = values[-24]
        lag_168h = values[0]
        roll_mean_24h = float(np.mean(values[-24:]))
        roll_mean_168h = float(np.mean(values))

        hour = target_ts.hour
        dayofweek = target_ts.weekday()
        month = target_ts.month
        year = target_ts.year
        is_weekend = int(dayofweek >= 5)
        dayofyear = target_ts.timetuple().tm_yday

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        doy_sin = np.sin(2 * np.pi * dayofyear / 365.25)
        doy_cos = np.cos(2 * np.pi * dayofyear / 365.25)

        feat = {
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "doy_sin": doy_sin, "doy_cos": doy_cos,
            "dayofweek": dayofweek, "is_weekend": is_weekend,
            "month": month, "year": year,
            "lag_1h": lag_1h, "lag_24h": lag_24h, "lag_168h": lag_168h,
            "roll_mean_24h": roll_mean_24h, "roll_mean_168h": roll_mean_168h,
        }
        return feat, hour
    finally:
        if own_conn:
            conn.close()


def forecast(target_ts: datetime, grid: str = "Cebu", z: float = 1.645, conn=None):
    """
    Returns expected demand + confidence band for target_ts.
    z=1.645 -> ~90% interval; z=1.96 -> ~95% interval.
    """
    model, bands, feature_cols = _load_artifacts()
    feat, hour = build_features(target_ts, grid=grid, conn=conn)

    X = np.array([[feat[c] for c in feature_cols]])
    expected = float(model.predict(X)[0])

    std = bands.get(str(hour), {}).get("std", 0.0)
    lower = expected - z * std
    upper = expected + z * std

    return {
        "timestamp": target_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "grid": grid,
        "expected_mw": round(expected, 2),
        "lower_bound_mw": round(lower, 2),
        "upper_bound_mw": round(upper, 2),
        "confidence_level": "90%" if abs(z - 1.645) < 1e-6 else f"z={z}",
        "model_version": MODEL_VERSION,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
