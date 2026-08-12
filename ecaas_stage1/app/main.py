"""
ECAAS Stage 1 — Forecast Service API

Run:
    cd app && uvicorn main:app --reload --port 8000

Try:
    GET /forecast?timestamp=2025-06-15T14:00:00&grid=Cebu
"""
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query

from db import get_connection, init_db
from forecast_service import forecast, InsufficientHistoryError

app = FastAPI(
    title="ECAAS Forecast Service",
    description="Stage 1: expected demand + confidence band on demand, "
                 "backed by the Cebu hourly load history.",
    version="0.1.0",
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def get_forecast(
    timestamp: str = Query(..., description="ISO timestamp, e.g. 2025-06-15T14:00:00"),
    grid: str = Query("Cebu", description="Grid/area name"),
):
    try:
        target_ts = datetime.fromisoformat(timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be ISO 8601, e.g. 2025-06-15T14:00:00")

    conn = get_connection()
    try:
        # serve from cache if we've already computed this exact hour
        cached = conn.execute(
            "SELECT * FROM forecast_cache WHERE timestamp = ? AND grid = ?",
            (target_ts.strftime("%Y-%m-%d %H:%M:%S"), grid),
        ).fetchone()
        if cached:
            return {
                "timestamp": cached["timestamp"],
                "grid": cached["grid"],
                "expected_mw": cached["expected_mw"],
                "lower_bound_mw": cached["lower_bound_mw"],
                "upper_bound_mw": cached["upper_bound_mw"],
                "model_version": cached["model_version"],
                "generated_at": cached["generated_at"],
                "cached": True,
            }

        try:
            result = forecast(target_ts, grid=grid, conn=conn)
        except InsufficientHistoryError as e:
            raise HTTPException(status_code=422, detail=str(e))

        conn.execute(
            """
            INSERT OR REPLACE INTO forecast_cache
            (timestamp, grid, expected_mw, lower_bound_mw, upper_bound_mw, model_version, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["timestamp"], result["grid"], result["expected_mw"],
                result["lower_bound_mw"], result["upper_bound_mw"],
                result["model_version"], result["generated_at"],
            ),
        )
        conn.commit()
        result["cached"] = False
        return result
    finally:
        conn.close()
