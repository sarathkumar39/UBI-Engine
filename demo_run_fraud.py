import sys
import json
from fraud import find_fraud_from_csv


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/demo_trips_with_geo.csv'
    alerts = find_fraud_from_csv(path)
    print(json.dumps(alerts, indent=2))


if __name__ == '__main__':
    main()
