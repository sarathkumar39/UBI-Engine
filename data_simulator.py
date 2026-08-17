"""
Simple synthetic telematics trip simulator.
Generates a CSV with columns:
 driver_id,trip_id,start_ts,duration_sec,distance_km,avg_speed_kmh,hard_brakes,rapid_accels,start_hour,weekday,risk_score,label

This script uses only the Python standard library so it can run without installing packages.
"""
import argparse
import csv
import math
import random
import uuid
from datetime import datetime, timedelta


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def gen_trip_with_params(driver_id, device_id, start_ts, driver_home, trip_type='commute'):
    trip_id = str(uuid.uuid4())
    # duration: commute shorter, long_haul longer
    if trip_type == 'commute':
        duration = random.randint(5*60, 45*60)
        avg_speed = round(random.uniform(20, 80), 1)
    else:
        duration = random.randint(30*60, 6*60*60)
        avg_speed = round(random.uniform(40, 120), 1)
    distance = round(avg_speed * (duration / 3600.0), 2)
    # safety events
    hard_brakes = random.choices([0,1,2,3,4,5], weights=[65,18,8,5,3,1])[0]
    rapid_accels = random.choices([0,1,2,3,4,5], weights=[72,14,8,4,1,1])[0]
    start_hour = start_ts.hour
    weekday = start_ts.weekday()  # 0=Mon

    # geolocation: start near driver's home (small jitter)
    base_lat, base_lon = driver_home
    # commutes start near home; long_haul may start from home or hub
    if trip_type == 'commute':
        start_lat = base_lat + random.uniform(-0.01, 0.01)
        start_lon = base_lon + random.uniform(-0.01, 0.01)
    else:
        start_lat = base_lat + random.uniform(-0.05, 0.05)
        start_lon = base_lon + random.uniform(-0.05, 0.05)

    # pick random bearing and compute end coords based on distance
    R = 6371.0
    bearing = random.uniform(0, 2*math.pi)
    ang_dist = distance / R
    lat1 = math.radians(start_lat)
    lon1 = math.radians(start_lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(ang_dist) + math.cos(lat1)*math.sin(ang_dist)*math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing)*math.sin(ang_dist)*math.cos(lat1), math.cos(ang_dist)-math.sin(lat1)*math.sin(lat2))
    end_lat = math.degrees(lat2)
    end_lon = math.degrees(lon2)

    # end timestamp
    end_ts = start_ts + timedelta(seconds=duration)

    # risk scoring
    score = 0.6 * hard_brakes + 0.5 * rapid_accels + 0.02 * max(0, avg_speed - 80)
    if start_hour >= 22 or start_hour <= 5:
        score += 0.5
    prob = sigmoid(score - 1.5)
    label = 1 if random.random() < prob else 0

    return {
        'driver_id': driver_id,
        'device_id': device_id,
        'trip_id': trip_id,
        'trip_type': trip_type,
        'start_ts': start_ts.isoformat(),
        'end_ts': end_ts.isoformat(),
        'duration_sec': duration,
        'distance_km': distance,
        'avg_speed_kmh': avg_speed,
        'hard_brakes': hard_brakes,
        'rapid_accels': rapid_accels,
        'start_hour': start_hour,
        'weekday': weekday,
        'start_lat': round(start_lat, 6),
        'start_lon': round(start_lon, 6),
        'end_lat': round(end_lat, 6),
        'end_lon': round(end_lon, 6),
        'risk_score': round(prob, 4),
        'label': label,
    }


def gen_trip(driver_id, base_dt, driver_home):
    # legacy helper: choose random start time and device and make a commute
    start_offset = random.randint(0, 7 * 24 * 60 * 60)
    start_ts = base_dt + timedelta(seconds=start_offset)
    device_id = f"dev_{driver_id}_01"
    return gen_trip_with_params(driver_id, device_id, start_ts, driver_home, 'commute')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic telematics trips CSV')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    parser.add_argument('--n-drivers', type=int, default=50)
    parser.add_argument('--n-trips', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    drivers = [f"driver_{i+1:03d}" for i in range(args.n_drivers)]
    base_dt = datetime.now() - timedelta(days=7)

    # assign a home/base location per driver from a small set of city hubs
    hubs = [
        (12.9715987, 77.594566),   # Bangalore
        (19.0759837, 72.8776559),  # Mumbai
        (28.6139391, 77.2090212),  # Delhi
        (13.0826802, 80.2707184),  # Chennai
    ]
    driver_homes = {d: random.choice(hubs) for d in drivers}

    fieldnames = ['driver_id','device_id','trip_id','trip_type','start_ts','end_ts','duration_sec','distance_km','avg_speed_kmh','hard_brakes','rapid_accels','start_hour','weekday','start_lat','start_lon','end_lat','end_lon','risk_score','label']

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(args.n_trips):
            driver_id = random.choice(drivers)
            row = gen_trip(driver_id, base_dt, driver_homes[driver_id])
            writer.writerow(row)

    print(f"Wrote {args.n_trips} trips for {len(drivers)} drivers to: {args.output}")


if __name__ == '__main__':
    main()
