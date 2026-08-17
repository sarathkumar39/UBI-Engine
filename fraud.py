"""
Fraud detection rules for the UBI POC with configurable thresholds and an
impossible-journey detector using lat/lon + Haversine distance/time checks.

Rules implemented:
 - high_speed: avg_speed_kmh > config.high_speed_threshold_kmh
 - short_aggressive: duration_sec < config.short_trip_duration_sec and (hard_brakes + rapid_accels) >= config.short_trip_events
 - rapid_consecutive: consecutive trips with gap < config.rapid_consecutive_gap_sec and distance > config.rapid_consecutive_min_distance_km
 - impossible_journey: implied speed between prev end location and curr start location exceeds config.max_implied_speed_kmh

Returns a list of alerts: {trip_id, driver_id, reason, detail}
"""
import os
import json
import math
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'fraud_config.json')

DEFAULT_CONFIG = {
    "high_speed_threshold_kmh": 160,
    "short_trip_duration_sec": 120,
    "short_trip_events": 3,
    "rapid_consecutive_gap_sec": 60,
    "rapid_consecutive_min_distance_km": 1.0,
    "max_implied_speed_kmh": 220
}


def load_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            # ensure defaults
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
    except Exception:
        pass
    return DEFAULT_CONFIG.copy()


def haversine_km(lat1, lon1, lat2, lon2):
    # Returns distance in kilometers between two points
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_fraud_from_csv(csv_path: str, driver_id: str = None) -> List[Dict]:
    df = pd.read_csv(csv_path)
    return find_fraud(df, driver_id)


def find_fraud(df: pd.DataFrame, driver_id: str = None) -> List[Dict]:
    cfg = load_config()
    alerts = []
    df = df.copy()

    # parse timestamps
    if 'start_ts' in df.columns:
        df['start_ts_parsed'] = pd.to_datetime(df['start_ts'], errors='coerce')
    else:
        df['start_ts_parsed'] = pd.NaT

    if 'end_ts' in df.columns:
        df['end_ts_parsed'] = pd.to_datetime(df['end_ts'], errors='coerce')
    else:
        # try to approximate end_ts as start_ts + duration_sec
        if 'duration_sec' in df.columns:
            df['end_ts_parsed'] = df.apply(lambda r: pd.to_datetime(r['start_ts'], errors='coerce') + pd.to_timedelta(r['duration_sec'], unit='s') if pd.notna(r.get('start_ts')) else pd.NaT, axis=1)
        else:
            df['end_ts_parsed'] = pd.NaT

    # filter by driver if requested
    if driver_id is not None and 'driver_id' in df.columns:
        df = df[df['driver_id'] == driver_id]

    # rule 1: high speed
    if 'avg_speed_kmh' in df.columns:
        high_speed = df[df['avg_speed_kmh'] > cfg.get('high_speed_threshold_kmh', DEFAULT_CONFIG['high_speed_threshold_kmh'])]
        for _, r in high_speed.iterrows():
            alerts.append({
                'trip_id': r.get('trip_id'),
                'driver_id': r.get('driver_id'),
                'reason': 'high_speed',
                'detail': f"avg_speed_kmh={r.get('avg_speed_kmh')}"
            })

    # rule 2: short duration with many events
    if 'duration_sec' in df.columns and 'hard_brakes' in df.columns and 'rapid_accels' in df.columns:
        short_aggr = df[(df['duration_sec'] < cfg.get('short_trip_duration_sec', DEFAULT_CONFIG['short_trip_duration_sec'])) & ((df['hard_brakes'] + df['rapid_accels']) >= cfg.get('short_trip_events', DEFAULT_CONFIG['short_trip_events']))]
        for _, r in short_aggr.iterrows():
            alerts.append({
                'trip_id': r.get('trip_id'),
                'driver_id': r.get('driver_id'),
                'reason': 'short_aggressive',
                'detail': f"duration_sec={r.get('duration_sec')}, events={r.get('hard_brakes') + r.get('rapid_accels')}"
            })

    # rule 3: rapid consecutive trips
    if 'start_ts_parsed' in df.columns and 'distance_km' in df.columns and 'driver_id' in df.columns:
        df_sorted = df.sort_values(['driver_id', 'start_ts_parsed'])
        grouped = df_sorted.groupby('driver_id')
        for driver, g in grouped:
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                prev = g.loc[i-1]
                curr = g.loc[i]
                if pd.isna(prev['start_ts_parsed']) or pd.isna(curr['start_ts_parsed']):
                    continue
                time_diff = (curr['start_ts_parsed'] - prev['start_ts_parsed']).total_seconds()
                if time_diff >= 0 and time_diff < cfg.get('rapid_consecutive_gap_sec', DEFAULT_CONFIG['rapid_consecutive_gap_sec']) and curr.get('distance_km', 0) > cfg.get('rapid_consecutive_min_distance_km', DEFAULT_CONFIG['rapid_consecutive_min_distance_km']):
                    alerts.append({
                        'trip_id': curr.get('trip_id'),
                        'driver_id': driver,
                        'reason': 'rapid_consecutive_trip',
                        'detail': f"time_diff_s={time_diff}, distance_km={curr.get('distance_km')}"
                    })

    # rule 4: impossible journey (geospatial)
    # requires prev end location and curr start location: columns may be named start_lat/start_lon and end_lat/end_lon
    lat_cols = ('start_lat', 'start_lon', 'end_lat', 'end_lon')
    has_geo = all(c in df.columns for c in lat_cols)
    if has_geo and 'driver_id' in df.columns:
        df_sorted = df.sort_values(['driver_id', 'start_ts_parsed'])
        grouped = df_sorted.groupby('driver_id')
        for driver, g in grouped:
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                prev = g.loc[i-1]
                curr = g.loc[i]
                # get prev end coords
                try:
                    prev_lat = float(prev.get('end_lat'))
                    prev_lon = float(prev.get('end_lon'))
                    curr_lat = float(curr.get('start_lat'))
                    curr_lon = float(curr.get('start_lon'))
                except Exception:
                    continue
                # compute distance
                distance_km = haversine_km(prev_lat, prev_lon, curr_lat, curr_lon)
                # compute time diff between prev end and curr start
                if pd.isna(prev.get('end_ts_parsed')) or pd.isna(curr.get('start_ts_parsed')):
                    continue
                time_diff_s = (curr['start_ts_parsed'] - prev['end_ts_parsed']).total_seconds()
                # avoid division by zero or negative gaps
                if time_diff_s <= 0:
                    # overlapping or impossible timestamps
                    alerts.append({
                        'trip_id': curr.get('trip_id'),
                        'driver_id': driver,
                        'reason': 'impossible_journey',
                        'detail': f"negative_or_zero_gap, distance_km={distance_km:.2f}, gap_s={time_diff_s}"
                    })
                    continue
                implied_speed_kmh = (distance_km / (time_diff_s / 3600.0)) if time_diff_s > 0 else float('inf')
                if implied_speed_kmh > cfg.get('max_implied_speed_kmh', DEFAULT_CONFIG['max_implied_speed_kmh']):
                    alerts.append({
                        'trip_id': curr.get('trip_id'),
                        'driver_id': driver,
                        'reason': 'impossible_journey',
                        'detail': f"distance_km={distance_km:.2f}, gap_s={time_diff_s:.1f}, implied_speed_kmh={implied_speed_kmh:.1f}"
                    })

    # aggregate alerts per trip and compute a severity score
    per_trip = {}
    # simple scoring contributions per rule (tunable)
    def contribution_for_rule(rule, detail_val=None):
        if rule == 'high_speed':
            return 30
        if rule == 'short_aggressive':
            return 15
        if rule == 'rapid_consecutive_trip':
            return 20
        if rule == 'impossible_journey':
            # if detail contains implied_speed_kmh, scale contribution
            if detail_val and 'implied_speed_kmh' in detail_val:
                try:
                    v = float(detail_val['implied_speed_kmh'])
                    base = min(80, int((v / cfg.get('max_implied_speed_kmh', DEFAULT_CONFIG['max_implied_speed_kmh'])) * 80))
                    return max(40, base)
                except Exception:
                    return 60
            return 60
        return 10

    for a in alerts:
        tid = a.get('trip_id') or a.get('trip') or 'unknown'
        if tid not in per_trip:
            per_trip[tid] = {
                'trip_id': tid,
                'driver_id': a.get('driver_id'),
                'reasons': [],
                'evidence': {},
                'fraud_score': 0,
            }
        reason = a.get('reason')
        detail = a.get('detail')
        per_trip[tid]['reasons'].append({'rule': reason, 'detail': detail})
        # try to pass numeric detail to contribution calculator
        detail_val = {}
        try:
            # parse implied_speed_kmh if present in detail text
            if detail and 'implied_speed_kmh=' in detail:
                parts = detail.split(',')
                for p in parts:
                    if 'implied_speed_kmh' in p:
                        k, v = p.split('=')
                        detail_val['implied_speed_kmh'] = float(v)
        except Exception:
            pass
        contrib = contribution_for_rule(reason, detail_val)
        per_trip[tid]['fraud_score'] += contrib
        # collect evidence
        per_trip[tid]['evidence'].update({reason: detail})

    # finalize scores and severities
    results = []
    for tid, info in per_trip.items():
        score = min(100, int(info['fraud_score']))
        if score >= 70:
            sev = 'high'
        elif score >= 40:
            sev = 'medium'
        else:
            sev = 'low'
        results.append({
            'trip_id': info['trip_id'],
            'driver_id': info.get('driver_id'),
            'fraud_score': score,
            'severity': sev,
            'reasons': info['reasons'],
            'evidence': info['evidence']
        })

    return results


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/trips.csv'
    alerts = find_fraud_from_csv(path)
    print(f'Found {len(alerts)} alerts')
    for a in alerts[:20]:
        print(a)
