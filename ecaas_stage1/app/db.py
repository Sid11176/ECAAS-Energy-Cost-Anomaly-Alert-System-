"""
SQLite schema for ECAAS.

Two tables land here in Stage 1:
- readings: historical + incoming actual demand values (seeded from the
  cleaned Cebu CSV; Stage 2's POST /reading will insert into this same table)
- forecast_cache: memorizes forecast() calls so repeated hits for the same
  hour don't re-run the model unnecessarily

anomaly_log (Stage 2) and alerts (Stage 5) are intentionally NOT created
here — each stage owns its own table so the schema grows with the roadmap
instead of being pre-built speculatively.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ecaas.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    timestamp   TEXT PRIMARY KEY,   -- ISO 'YYYY-MM-DD HH:MM:SS', hourly
    grid        TEXT NOT NULL,
    demand_mw   REAL NOT NULL,
    source      TEXT NOT NULL DEFAULT 'historical'  -- 'historical' | 'live'
);

CREATE INDEX IF NOT EXISTS idx_readings_grid_ts ON readings(grid, timestamp);

CREATE TABLE IF NOT EXISTS forecast_cache (
    timestamp       TEXT NOT NULL,
    grid            TEXT NOT NULL,
    expected_mw     REAL NOT NULL,
    lower_bound_mw  REAL NOT NULL,
    upper_bound_mw  REAL NOT NULL,
    model_version   TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    PRIMARY KEY (timestamp, grid)
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")
