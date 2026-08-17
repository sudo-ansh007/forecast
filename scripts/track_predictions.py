#!/usr/bin/env python3
"""Track predictions over time. Save timestamped predictions, measure later vs actual.

Workflow:
  1. Weekly: python track_predictions.py --save
     Saves current predictions to tracked_predictions/ with timestamp

  2. After deals close (90d later): python track_predictions.py --validate 2026-05-20
     Queries which deals from that date actually closed
     Compares predicted vs actual
     Auto-alerts if conversion rate missed target

Usage:
    python track_predictions.py --save              # save today's predictions
    python track_predictions.py --validate 2026-05-20  # measure from that date
    python track_predictions.py --report           # show all tracked runs
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import csv

TRACKING_DIR = Path("models") / "tracked_predictions"
TRACKING_DIR.mkdir(exist_ok=True)
VALIDATION_SUMMARY = Path("models") / "validation_summary.csv"


def save_predictions():
    """Save current open_pipeline_predictions.csv with timestamp."""
    csv_path = Path("results") / "open_pipeline_predictions.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run predict.py first.")
        return False

    df = pd.read_csv(csv_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tracked_file = TRACKING_DIR / f"predictions_{timestamp}.csv"

    # Add save date to tracking
    df["prediction_date"] = datetime.now().date()
    df["prediction_timestamp"] = timestamp

    df.to_csv(tracked_file, index=False)
    print(f"✓ Saved {len(df)} predictions to {tracked_file}")

    # Log in index
    with open(TRACKING_DIR / "index.txt", "a") as f:
        f.write(f"{timestamp}\t{len(df)} deals\n")

    return True


def validate_predictions(from_date_str):
    """Measure predictions from from_date_str against actual outcomes.

    Query: which deals predicted on from_date_str actually closed?
    Measure: actual close rate vs predicted average probability
    """
    from_date = datetime.strptime(from_date_str, "%Y-%m-%d").date()

    # Find prediction file from that date
    pred_files = sorted(TRACKING_DIR.glob("predictions_*.csv"))
    pred_file = None
    for f in pred_files:
        file_date = datetime.strptime(
            f.stem.split("_", 1)[1], "%Y-%m-%d_%H%M%S"
        ).date()
        if file_date == from_date:
            pred_file = f
            break

    if not pred_file:
        print(f"No prediction file found for {from_date_str}")
        return False

    predictions = pd.read_csv(pred_file)
    print(f"Loaded {len(predictions)} predictions from {pred_file.name}")

    # Load training data (source of truth for outcomes)
    train = pd.read_parquet(Path("dataset") / "train.parquet")
    train["display_id"] = train["display_id"].astype(str)

    # Merge: which predicted deals are now in train.parquet (closed)?
    predictions["display_id"] = predictions["display_id"].astype(str)
    outcomes = predictions.merge(
        train[["display_id", "is_won"]], on="display_id", how="left"
    )

    # Deals that closed (in outcomes.is_won == 1)
    closed = outcomes[outcomes.is_won == 1]
    if len(closed) == 0:
        print(f"No deals from {from_date_str} have closed yet.")
        return False

    # Metrics by signal_strength
    print(f"\nValidation: {len(closed)} deals closed from {len(predictions)} predicted")
    print(f"{'Signal':<10} {'Closed':>8} {'Actual %':>12} {'Predicted %':>14} {'Error':>8}")
    print("-" * 60)

    summary = {}
    for sig in ["high", "medium", "low", "none"]:
        sub = closed[closed.signal_strength == sig]
        if len(sub) == 0:
            continue
        actual_rate = sub.is_won.mean()
        pred_rate = sub.win_probability.mean()
        error = actual_rate - pred_rate
        summary[sig] = {
            "closed": len(sub),
            "actual_rate": actual_rate,
            "pred_rate": pred_rate,
            "error": error,
        }
        print(
            f"{sig:<10} {len(sub):>8d} {actual_rate:>11.1%} {pred_rate:>13.1%} {error:>+7.1%}"
        )

    # Overall calibration
    overall_actual = closed.is_won.mean()
    overall_pred = closed.win_probability.mean()
    overall_error = overall_actual - overall_pred
    print("-" * 60)
    print(f"{'OVERALL':<10} {len(closed):>8d} {overall_actual:>11.1%} {overall_pred:>13.1%} {overall_error:>+7.1%}")

    # Alert if gap > 20%
    if abs(overall_error) > 0.20:
        print(f"\n🚨 ALERT: Model error {overall_error:+.1%} exceeds 20% threshold")
        print("   Check for cohort shift or feature drift")
    elif abs(overall_error) > 0.10:
        print(f"\n⚠ WARNING: Model error {overall_error:+.1%} (monitor)")
    else:
        print(f"\n✓ OK: Model error {overall_error:+.1%} (within acceptable range)")

    # Save validation result
    val_result = {
        "prediction_date": from_date_str,
        "deals_predicted": len(predictions),
        "deals_closed": len(closed),
        "actual_close_rate": overall_actual,
        "predicted_close_rate": overall_pred,
        "error": overall_error,
        "validation_date": datetime.now().date(),
    }
    _log_validation(val_result)

    return True


def report():
    """Show all tracked predictions and validations."""
    pred_files = sorted(TRACKING_DIR.glob("predictions_*.csv"))
    print(f"\nTracked Predictions ({len(pred_files)} files):")
    for f in pred_files:
        size = pd.read_csv(f).shape[0]
        print(f"  {f.name}: {size} deals")

    if VALIDATION_SUMMARY.exists():
        print(f"\nValidation History:")
        results = pd.read_csv(VALIDATION_SUMMARY)
        for _, row in results.iterrows():
            print(
                f"  {row['prediction_date']}: {row['deals_closed']} closed, "
                f"error {row['error']:+.1%}"
            )


def _log_validation(result):
    """Append validation result to summary CSV."""
    exists = VALIDATION_SUMMARY.exists()
    with open(VALIDATION_SUMMARY, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(result)
    print(f"✓ Logged to {VALIDATION_SUMMARY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--save",
        action="store_true",
        help="Save today's predictions to tracking",
    )
    ap.add_argument(
        "--validate",
        metavar="DATE",
        help="Validate predictions from DATE (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Show all tracked runs and validations",
    )
    args = ap.parse_args()

    if args.save:
        save_predictions()
    elif args.validate:
        validate_predictions(args.validate)
    elif args.report:
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
