"""
Test scoring script for the UBI POC Day 1.
Reads data/sample_request.json and computes the same rule-based risk probability
used by data_simulator.py, printing results to the console.
"""
import json
import math
from pathlib import Path


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def score_trip(t):
    score = 0.6 * t.get('hard_brakes', 0) + 0.5 * t.get('rapid_accels', 0) + 0.02 * max(0, t.get('avg_speed_kmh', 0) - 80)
    sh = t.get('start_hour', 0)
    if sh >= 22 or sh <= 5:
        score += 0.5
    prob = sigmoid(score - 1.5)
    return round(prob, 4)


def main():
    p = Path('data') / 'sample_request.json'
    if not p.exists():
        print('Missing', p)
        return
    req = json.loads(p.read_text())
    out = []
    for t in req.get('trips', []):
        prob = score_trip(t)
        out.append({'driver_id': t.get('driver_id'), 'probability': prob, 'inputs': t})
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
