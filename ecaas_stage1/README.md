# ECAAS — Stage 1: Reuse & Reshape Forecast Service

Trains an XGBoost demand forecaster on the cleaned Cebu hourly load history
(2013–2025) and exposes it as `GET /forecast`, returning expected demand +
a confidence band for any hour — the foundation Stage 2's deviation engine
will call.

## Setup

```bash
pip install -r requirements.txt

# 1. Train the model (writes to models/)
python train_forecast_model.py

# 2. Seed the readings table from the cleaned CSV
cd app && python seed_readings.py

# 3. Run the API
uvicorn main:app --reload --port 8000
```

## Try it

```bash
curl "http://127.0.0.1:8000/forecast?timestamp=2025-06-15T14:00:00&grid=Cebu"
```

```json
{
  "timestamp": "2025-06-15 14:00:00",
  "grid": "Cebu",
  "expected_mw": 877.46,
  "lower_bound_mw": 843.66,
  "upper_bound_mw": 911.27,
  "confidence_level": "90%",
  "model_version": "xgb_v1_2013_2025",
  "generated_at": "...",
  "cached": false
}
```

## Design notes

- **Features**: cyclical hour-of-day and day-of-year encodings, day-of-week,
  weekend flag, and three lag windows (1h, 24h, 168h) plus rolling means
  (24h, 168h). All lags are strictly backward-looking — no leakage.
- **Confidence band**: not quantile regression — residual std is bucketed
  by hour-of-day from a 180-day time-based holdout (never shuffled), then
  used as `expected ± z·std`. Ramp hours (7–9am) are visibly noisier than
  overnight hours, which the per-hour bucketing captures.
- **Holdout performance**: ~1.5% MAPE / ~11 MW MAE on the last 180 days.
  Good enough that the deviation engine in Stage 2 won't be fighting
  forecast noise — the interesting anomalies will be real ones.
- **SQLite schema**: `readings` and `forecast_cache` only. `anomaly_log`
  (Stage 2) and `alerts` (Stage 5) are deliberately not created yet — each
  stage owns its own table as it's built, rather than a schema designed
  speculatively up front.

## Known caveat — read before Stage 2

Requesting a forecast **far beyond the latest available reading** (e.g.
a date months past the last row in `readings`) still returns a result —
it silently falls back to the most recent 168h of actuals as lag inputs.
That's correct behavior for "next hour" or "next few days" forecasting,
but it means the endpoint doesn't yet distinguish between an interpolated
forecast and one built on stale lags. If you're demoing this live and
someone asks for a forecast far past your data's actual end date, be
ready to explain that — it's a known simplification, not a bug you missed.
A version flag (`extrapolated: true/false`) would be a clean Stage 2
add if it comes up.
