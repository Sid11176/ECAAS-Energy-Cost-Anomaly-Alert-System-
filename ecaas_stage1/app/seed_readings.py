"""
One-time load: cleaned Cebu CSV -> readings table.
Everything loaded this way is tagged source='historical'.
Stage 2's POST /reading will insert new rows with source='live'.
"""
import csv
from pathlib import Path
from db import get_connection, init_db

CSV_PATH = Path(__file__).parent.parent / "data" / "cebu_hourly_demand_2013_2025_long.csv"


def seed():
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = [(r["timestamp"], r["grid"], float(r["demand_mw"]), "historical") for r in reader]

    cur.executemany(
        "INSERT OR REPLACE INTO readings (timestamp, grid, demand_mw, source) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    count = cur.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    print(f"Seeded {len(rows)} rows. Table now has {count} rows.")
    conn.close()


if __name__ == "__main__":
    seed()
