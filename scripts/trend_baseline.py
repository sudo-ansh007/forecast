#!/usr/bin/env python3
"""Trend baseline: ARIMA on historical closed deals.

When you had $X open pipeline, what did you close?
Answers: is the $7.2M forecast sane, or is the model drifting?

Usage:
    python trend_baseline.py                    # fit and plot
    python trend_baseline.py --forecast-only    # load saved model, forecast next month
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

DATASET_DIR = Path("dataset")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
except ImportError:
    print("WARNING: statsmodels not installed. Trend baseline requires: pip install statsmodels")
    ARIMA = None

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
BASELINE_FILE = OUT_DIR / "arima_baseline.pkl"


def prepare_monthly_actuals():
    """Aggregate closed deals into monthly revenue.

    dataset/train.parquet has closed_date (when deal was won). Bucket by month, sum ACV.
    Returns: (dates, actuals) — Series indexed by month-end.
    """
    train = pd.read_parquet(DATASET_DIR / "train.parquet")
    won = train[train.is_won == 1].copy()

    # Parse closed_date (may not exist; fallback to created_date proxy)
    if "closed_date" in won.columns:
        won["month"] = pd.to_datetime(won.closed_date).dt.to_period("M")
    else:
        # Proxy: created_date + days_in_sales_cycle
        won["resolved"] = pd.to_datetime(won.created_date) + pd.to_timedelta(
            won.days_in_sales_cycle, unit="D"
        )
        won["month"] = won.resolved.dt.to_period("M")

    monthly = won.groupby("month").agg(
        count=("acv", "count"), revenue=("acv", "sum")
    ).reset_index()
    monthly["month"] = monthly.month.dt.to_timestamp()
    monthly = monthly.set_index("month").sort_index()

    print(f"{len(monthly)} months in history")
    print(f"Revenue range: ${monthly.revenue.min()/1e6:.2f}M – ${monthly.revenue.max()/1e6:.2f}M")
    print(f"Mean: ${monthly.revenue.mean()/1e6:.2f}M/mo")
    return monthly.revenue


def fit_arima(actuals, order=(1, 1, 1)):
    """Fit ARIMA to monthly closed revenue. Return model and diagnostics."""
    if len(actuals) < 6:
        print(f"WARNING: only {len(actuals)} months; ARIMA unreliable with <12")

    model = ARIMA(actuals, order=order)
    result = model.fit()
    return result


def forecast_next_n(result, n_months=1):
    """Forecast next n months."""
    forecast = result.get_forecast(steps=n_months)
    return forecast.predicted_mean, forecast.conf_int()


def plot_diagnostics(result, actuals):
    """Plot ARIMA fit + forecast."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Fit
    ax = axes[0, 0]
    ax.plot(actuals.index, actuals, "o-", label="actual", color="#0072B2")
    ax.plot(result.fittedvalues.index, result.fittedvalues, "-", label="fit",
            color="#E69F00", lw=2)
    ax.set_ylabel("Revenue ($)")
    ax.set_title("ARIMA Fit")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Residuals
    ax = axes[0, 1]
    ax.plot(result.resid, "o-", color="#999")
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    ax.grid(alpha=0.3)

    # ACF
    ax = axes[1, 0]
    plot_acf(result.resid, lags=10, ax=ax)
    ax.set_title("ACF (residuals should be white noise)")

    # PACF
    ax = axes[1, 1]
    plot_pacf(result.resid, lags=10, ax=ax)
    ax.set_title("PACF")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "arima_diagnostics.png", dpi=100, bbox_inches="tight")
    print(f"Saved diagnostics to {OUT_DIR}/arima_diagnostics.png")
    plt.close()


def plot_forecast(result, actuals, n_months=3):
    """Plot history + forecast."""
    forecast_mu, forecast_ci = forecast_next_n(result, n_months)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actuals.index, actuals, "o-", label="actual closed", color="#0072B2", lw=2)
    ax.plot(result.fittedvalues.index, result.fittedvalues, "--", label="ARIMA fit",
            color="#E69F00", lw=1.5, alpha=0.7)

    x_future = forecast_mu.index
    ax.plot(x_future, forecast_mu, "s-", label="forecast", color="#D55E00", lw=2, ms=8)
    ax.fill_between(x_future, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                    color="#D55E00", alpha=0.2, label="90% CI")

    ax.set_ylabel("Monthly Closed Revenue ($)")
    ax.set_title("Revenue Trend: Historical vs Forecast")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "arima_forecast.png", dpi=100, bbox_inches="tight")
    print(f"Saved forecast plot to {OUT_DIR}/arima_forecast.png")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-only", action="store_true",
                    help="Load saved model, skip refitting")
    args = ap.parse_args()

    if ARIMA is None:
        print("statsmodels required. Install: pip install statsmodels")
        return

    if args.forecast_only:
        if not BASELINE_FILE.exists():
            print(f"No saved baseline at {BASELINE_FILE}. Run without --forecast-only first.")
            return
        result = pickle.load(open(BASELINE_FILE, "rb"))
        forecast_mu, forecast_ci = forecast_next_n(result, n_months=1)
        print(f"\nForecast (next month):")
        print(f"  Mean: ${forecast_mu.values[0]/1e6:.2f}M")
        print(f"  90% CI: ${forecast_ci.iloc[0,0]/1e6:.2f}M – ${forecast_ci.iloc[0,1]/1e6:.2f}M")
        return

    print("Loading historical closed deals...")
    actuals = prepare_monthly_actuals()

    print("\nFitting ARIMA(1,1,1)...")
    result = fit_arima(actuals)

    print("\nARIMA Summary:")
    print(result.summary())

    print("\nModel diagnostics: ACF/PACF, residuals")
    plot_diagnostics(result, actuals)

    print("\nGenerating forecast...")
    forecast_mu, forecast_ci = forecast_next_n(result, n_months=3)
    print("Next 3 months forecast:")
    for i, mu in enumerate(forecast_mu.values):
        ci = forecast_ci.iloc[i]
        print(f"  Month +{i+1}: ${mu/1e6:.2f}M (90% CI: ${ci[0]/1e6:.2f}M – ${ci[1]/1e6:.2f}M)")

    plot_forecast(result, actuals, n_months=3)

    # Save for later reuse
    pickle.dump(result, open(BASELINE_FILE, "wb"))
    print(f"\nSaved ARIMA model to {BASELINE_FILE}")

    # Sanity check against model forecast
    print("\n" + "="*60)
    print("Sanity Check: Model Forecast vs Trend Baseline")
    print("="*60)
    model_forecast_total = 7.21e6  # $7.21M from open_pipeline_predictions.csv
    trend_forecast = forecast_mu.values[0]
    ratio = model_forecast_total / trend_forecast if trend_forecast > 0 else 0
    print(f"Model (Σ p×acv):  ${model_forecast_total/1e6:.2f}M (total open forecast)")
    print(f"Trend (ARIMA):     ${trend_forecast/1e6:.2f}M (next month baseline)")
    print(f"Ratio (model/trend): {ratio:.2f}x")
    if 0.8 <= ratio <= 1.2:
        print("✓ Aligned. Model forecast is sane.")
    elif ratio > 1.2:
        print("⚠ Model predicts higher than historical trend. Check for recent cohort shift.")
    else:
        print("⚠ Model predicts lower than trend. Possible drift in calibration or features.")


if __name__ == "__main__":
    main()
