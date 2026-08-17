#!/usr/bin/env python3
"""Backtest: score historical deals when they were open, check if forecast held.

Take deals that are now closed (dataset/train.parquet). For each deal, use the model to
score it as if it were open (using features as they were at creation time).
Compare predicted win prob to actual outcome by signal_strength bucket.

Usage:
    python backtest_validation.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from predict import WinProbabilityModel

DATASET_DIR = Path("dataset")

# Load closed deals (dataset/train.parquet has is_won labels)
train = pd.read_parquet(DATASET_DIR / "train.parquet")
train["created_date"] = pd.to_datetime(train.created_date)

# Focus on deals created 3–6 months before the latest deal in the set
# (old enough to have resolved, recent enough to be comparable)
latest = train.created_date.max()
cutoff_recent = latest - pd.Timedelta(days=90)
cutoff_old = latest - pd.Timedelta(days=180)

backtest = train[(train.created_date >= cutoff_old) & (train.created_date < cutoff_recent)].copy()
print(f"Backtest cohort: {len(backtest)} deals created {cutoff_old.date()} to {cutoff_recent.date()}")
print(f"  Actual wins: {backtest.is_won.sum()} ({backtest.is_won.mean():.1%})")

# Score with the current model
model = WinProbabilityModel()
p = model.predict_proba(backtest)

backtest["predicted_prob"] = p
backtest["signal_strength"] = pd.cut(
    p, [0, 0.1, 0.3, 0.5, 1.0], labels=["none", "low", "medium", "high"]
)
backtest["predicted_win"] = (p >= model.threshold).astype(int)

print(f"\nModel predictions on backtest cohort:")
print(f"  Mean probability: {p.mean():.3f}")
print(f"  Flagged as wins (≥0.3): {(p >= 0.3).sum()} deals")

# By signal strength: did the forecast hold?
print(f"\n{'Signal':<10} {'Count':>6} {'Actual Win %':>14} {'Pred Avg %':>12} {'Error':>8}")
print("-" * 60)

for sig in ["high", "medium", "low", "none"]:
    sub = backtest[backtest.signal_strength == sig]
    if len(sub) == 0:
        continue
    actual_rate = sub.is_won.mean()
    pred_rate = sub.predicted_prob.mean()
    error = actual_rate - pred_rate
    print(
        f"{sig:<10} {len(sub):>6d} {actual_rate:>13.1%} {pred_rate:>11.1%} {error:>+7.1%}"
    )

# Calibration: sum(p) / actual
total_pred = p.sum()
total_actual = backtest.is_won.sum()
cal_ratio = total_pred / total_actual if total_actual > 0 else 0
print(f"\nCalibration ratio (sum(p) / actual wins): {cal_ratio:.3f}")
print(f"  <1.30 OK | 1.30-1.60 WARN | >1.60 ALARM")
if cal_ratio < 1.30:
    print("  ✓ OK")
elif cal_ratio < 1.60:
    print("  ⚠ WARN: model over-predicting")
else:
    print("  🚨 ALARM: model significantly over-predicting")

# Summary verdict
print(f"\n{'='*60}")
print("Backtest Verdict")
print("="*60)

high_sub = backtest[backtest.signal_strength == "high"]
if len(high_sub) > 0:
    high_actual = high_sub.is_won.mean()
    high_pred = high_sub.predicted_prob.mean()
    print(f"High-signal (>0.5) deals closed at {high_actual:.1%} (model predicted {high_pred:.1%})")
    if abs(high_actual - high_pred) < 0.15:
        print("  ✓ Forecast held")
    else:
        print("  ⚠ Forecast was off")
else:
    print("High-signal (>0.5) deals: none in this cohort")

low_sub = backtest[backtest.signal_strength == "low"]
if len(low_sub) > 0:
    low_actual = low_sub.is_won.mean()
    low_pred = low_sub.predicted_prob.mean()
    print(f"Low-signal (0.1-0.3) deals closed at {low_actual:.1%} (model predicted {low_pred:.1%})")
    if abs(low_actual - low_pred) < 0.10:
        print("  ✓ Forecast held")
    else:
        print("  ⚠ Forecast was off")
else:
    print("Low-signal (0.1-0.3) deals: none in this cohort")

none_sub = backtest[backtest.signal_strength == "none"]
if len(none_sub) > 0:
    none_actual = none_sub.is_won.mean()
    none_pred = none_sub.predicted_prob.mean()
    print(f"No-signal (<0.1) deals closed at {none_actual:.1%} (model predicted {none_pred:.1%})")
    if abs(none_actual - none_pred) < 0.10:
        print("  ✓ Forecast held")
    else:
        print("  ⚠ Forecast was off")
else:
    print("No-signal (<0.1) deals: none in this cohort")

print(f"\nCalibration: {cal_ratio:.3f}x ('OK' if <1.30)")
if cal_ratio < 1.30:
    print("✓ Model is trustworthy on this cohort. Ready to use on live open deals.")
else:
    print("⚠ Model over-predicts. May need recalibration before relying on $7.21M forecast.")
