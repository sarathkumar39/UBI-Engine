"""
Simple FastAPI server to score trips using the trained model saved in artifacts/model.joblib.
POST /score accepts JSON: { "trips": [ {trip fields...}, ... ] }
Returns per-trip probabilities and aggregate summary.
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import math

app = FastAPI(title='UBI POC Scoring API')

MODEL_PATH = os.path.join('artifacts', 'model.joblib')
FEATURE_COLS = ['duration_sec','distance_km','avg_speed_kmh','hard_brakes','rapid_accels','start_hour','weekday','speed_per_10km','brake_accel_sum','is_night']

class Trip(BaseModel):
    driver_id: Optional[str]
    duration_sec: int
    distance_km: float
    avg_speed_kmh: float
    hard_brakes: int = 0
    rapid_accels: int = 0
    start_hour: int = 12
    weekday: int = 0

class ScoreRequest(BaseModel):
    trips: List[Trip]

class TripScore(BaseModel):
    driver_id: Optional[str]
    probability: float

@app.on_event('startup')
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'Model not found at {MODEL_PATH}. Run training first.')
    model = joblib.load(MODEL_PATH)

def prepare_df(trips: List[Trip]) -> pd.DataFrame:
    rows = [t.dict() for t in trips]
    df = pd.DataFrame(rows)
    # Derived features (same as training)
    df['speed_per_10km'] = df['avg_speed_kmh'] / 10.0
    df['brake_accel_sum'] = df['hard_brakes'] + df['rapid_accels']
    df['is_night'] = df['start_hour'].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    # Ensure columns exist in expected order
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[FEATURE_COLS]

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/score', response_model=List[TripScore])
def score(req: ScoreRequest):
    try:
        df = prepare_df(req.trips)
        probs = model.predict_proba(df)[:, 1]
        out = []
        for trip, p in zip(req.trips, probs):
            out.append(TripScore(driver_id=trip.driver_id, probability=round(float(p), 4)))
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pricing endpoint
class PricingRequest(BaseModel):
    # Accept either a precomputed risk_score (0-100) or trips to compute an implied score
    risk_score: Optional[float] = None
    monthly_mileage: Optional[float] = 1000.0
    is_ev: Optional[bool] = False
    safe_driver: Optional[bool] = False

@app.post('/pricing/calculate')
def pricing_calculate(req: PricingRequest):
    try:
        # compute risk_score if not provided: use model on a single trip if trips not provided here
        rs = req.risk_score
        if rs is None:
            # No trips provided in this simple endpoint — expect caller to pass risk_score
            raise HTTPException(status_code=400, detail='risk_score is required for this endpoint')

        # map risk score boundary: pricing.py expects higher is better (safer). If your risk_score is model prob (higher means riskier),
        # convert externally before calling this endpoint.
        from pricing import compute_premium
        breakdown = compute_premium(risk_score=rs, monthly_mileage=req.monthly_mileage or 0.0, is_ev=bool(req.is_ev), safe_driver=bool(req.safe_driver))
        return breakdown
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/pricing/from_trips')
def pricing_from_trips(score_req: ScoreRequest, monthly_mileage: Optional[float] = 1000.0, is_ev: Optional[bool] = False, safe_driver: Optional[bool] = False):
    """Compute pricing directly from trips by scoring them with the model, averaging, and calling compute_premium."""
    try:
        df = prepare_df(score_req.trips)
        probs = model.predict_proba(df)[:,1]
        avg_prob = float(probs.mean()) if len(probs)>0 else 0.0
        # convert to risk_score where higher is safer
        risk_score = round((1.0 - avg_prob) * 100.0, 2)
        from pricing import compute_premium
        breakdown = compute_premium(risk_score=risk_score, monthly_mileage=monthly_mileage or 0.0, is_ev=bool(is_ev), safe_driver=bool(safe_driver))
        # include inferred risk and avg_prob
        breakdown['inferred_risk_score'] = risk_score
        breakdown['avg_trip_prob'] = avg_prob
        return breakdown
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/billing/generate')
def billing_generate(payload: dict):
    """Generate invoice for driver. Payload options:
       - driver_id (required)
       - period (e.g., '2026-08')
       - risk_score OR trips
       - monthly_mileage, is_ev, safe_driver
    """
    try:
        driver_id = payload.get('driver_id')
        if not driver_id:
            raise HTTPException(status_code=400, detail='driver_id is required')
        period = payload.get('period') or datetime.utcnow().strftime('%Y-%m')
        # Determine premium breakdown
        if 'risk_score' in payload and payload.get('risk_score') is not None:
            rs = float(payload.get('risk_score'))
            from pricing import compute_premium
            pb = compute_premium(risk_score=rs, monthly_mileage=payload.get('monthly_mileage',0.0), is_ev=payload.get('is_ev',False), safe_driver=payload.get('safe_driver',False))
        elif 'trips' in payload:
            # use model to score trips
            trips = payload.get('trips')
            score_req = ScoreRequest(trips=trips)
            res = pricing_from_trips(score_req, monthly_mileage=payload.get('monthly_mileage',1000.0), is_ev=payload.get('is_ev',False), safe_driver=payload.get('safe_driver',False))
            # res is dict from compute_premium + extras
            # remove functions/objects
            pb = res
        else:
            raise HTTPException(status_code=400, detail='Either risk_score or trips must be provided')

        # generate invoice
        from billing import generate_invoice
        inv = generate_invoice(driver_id=driver_id, period=period, premium_breakdown=pb)
        return inv
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/fraud-alerts')
def fraud_alerts(driver_id: Optional[str] = None):
    try:
        # run fraud detection on data/trips.csv
        trips_path = os.path.join('data', 'trips.csv')
        if not os.path.exists(trips_path):
            return []
        from fraud import find_fraud_from_csv
        alerts = find_fraud_from_csv(trips_path, driver_id=driver_id)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
