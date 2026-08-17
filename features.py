"""
Feature engineering utilities for the UBI POC.
Contains a simple function to load the trips CSV and prepare features and labels for modeling.
"""
import pandas as pd


def load_and_prepare(csv_path: str):
    """Load CSV and return X (features DataFrame) and y (labels Series).

    Uses simple trip-level features already present in the CSV.
    """
    df = pd.read_csv(csv_path)

    # Ensure expected columns
    expected = ['duration_sec','distance_km','avg_speed_kmh','hard_brakes','rapid_accels','start_hour','weekday','label']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Create some derived features
    df['speed_per_10km'] = df['avg_speed_kmh'] / 10.0
    df['brake_accel_sum'] = df['hard_brakes'] + df['rapid_accels']
    # night flag
    df['is_night'] = df['start_hour'].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)

    feature_cols = ['duration_sec','distance_km','avg_speed_kmh','hard_brakes','rapid_accels','start_hour','weekday','speed_per_10km','brake_accel_sum','is_night']
    X = df[feature_cols].copy()
    y = df['label'].copy()
    return X, y


if __name__ == '__main__':
    # Quick smoke test when run directly
    X, y = load_and_prepare('data/trips.csv')
    print('Loaded X shape:', X.shape)
    print('Label distribution:\n', y.value_counts())
