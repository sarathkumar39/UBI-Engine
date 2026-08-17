"""
Train a simple sklearn model on the trip-level data and save artifacts.
Saves:
 - artifacts/model.joblib
 - artifacts/train_report.txt
 - artifacts/sample_predictions.csv

Usage:
 python train_model.py --input data\\trips.csv --out-model artifacts\\model.joblib
"""
import argparse
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

from features import load_and_prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV with trips')
    parser.add_argument('--out-model', required=True, help='Output path for model.joblib')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    X, y = load_and_prepare(args.input)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state, stratify=y)

    # Simple pipeline: scaler + logistic regression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])

    pipe.fit(X_train, y_train)

    # Predictions and metrics
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    creport = classification_report(y_test, y_pred)

    # Prepare artifacts dir
    out_dir = os.path.dirname(args.out_model)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save model
    joblib.dump(pipe, args.out_model)

    # Save a small training report
    report_path = os.path.join(out_dir, 'train_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Training report - {datetime.utcnow().isoformat()} UTC\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Model: LogisticRegression (StandardScaler)\n")
        f.write('\n')
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write('\nConfusion matrix:\n')
        f.write(str(cm) + '\n')
        f.write('\nClassification report:\n')
        f.write(creport + '\n')

    # Save sample predictions
    sample_out = os.path.join(out_dir, 'sample_predictions.csv')
    df_test = X_test.copy()
    df_test['label'] = y_test.values
    df_test['pred_prob'] = y_prob
    df_test['pred_label'] = y_pred
    df_test.to_csv(sample_out, index=False)

    print('Saved model to', args.out_model)
    print('Saved report to', report_path)
    print('Saved sample predictions to', sample_out)


if __name__ == '__main__':
    main()
