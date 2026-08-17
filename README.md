UBI POC — Day 1

This project contains a simple synthetic telematics simulator and later will include model training, API, and dashboard.

Quick start (Day 1)

1. Open PowerShell and change to the project folder:
   cd "C:\Users\SarathkumarR\OneDrive - Xebia\Coding\AI project"

2. (Optional) Create and activate a venv:
   python -m venv venv
   .\venv\Scripts\Activate.ps1

3. Run the simulator (no dependencies required):
   python data_simulator.py --output data\trips.csv --n-drivers 20 --n-trips 500 --seed 42

The generated CSV will be at data\trips.csv and contains columns: driver_id, device_id, trip_id, trip_type, start_ts, end_ts, duration_sec, distance_km, avg_speed_kmh, hard_brakes, rapid_accels, start_hour, weekday, start_lat, start_lon, end_lat, end_lon, risk_score, label

Automated demo: impossible-journey alert

A small automated example is included to show how the demo geo file triggers the impossible-journey detector.

1. Use the provided demo geo CSV (already in the repo):
   data/demo_trips_with_geo.csv

2. Run the quick demo script which runs the fraud scanner and prints alerts:
   python demo_run_fraud.py data/demo_trips_with_geo.csv

Expected output (trimmed) — you should see an impossible_journey alert for driver_002 with a high fraud score, for example:
{
  "trip_id": "TRIP-IMPOSS-1",
  "driver_id": "driver_002",
  "fraud_score": 80,
  "severity": "high",
  "reasons": [
    {"rule": "impossible_journey", "detail": "distance_km=... , gap_s=... , implied_speed_kmh=..."}
  ]
}

If you want to re-generate more realistic geo trips, run the simulator with:
   python data_simulator.py --output data/trips.csv --n-drivers 50 --n-trips 2000 --seed 42

This will include device_id, trip_type (commute | long_haul), and geographic coordinates to enable impossible-journey checks and replay-device scenarios.

Next steps (Day 2/Day 3): feature engineering, model training, FastAPI server, Streamlit dashboard.
