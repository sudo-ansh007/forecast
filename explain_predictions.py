#!/usr/bin/env python3
"""Extract SHAP feature importance for deal explanations."""

import pickle
import pandas as pd
import numpy as np
import shap
from pathlib import Path


def load_model_and_features():
    """Load trained CatBoost model and feature metadata."""
    with open("models/model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open("models/catboost_model.pkl", "rb") as f:
        model = pickle.load(f)

    return model, metadata["features"], metadata["categoricals"]


def get_shap_values(model, X, background_size=100):
    """Compute SHAP values (expensive, ~2min for 1,912 deals)."""
    print(f"Computing SHAP values for {len(X)} deals...")

    # Use TreeExplainer (native CatBoost support, much faster than KernelExplainer)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values, explainer.expected_value


def extract_top_factors(shap_vals, row_idx, feature_names, X_row, top_n=5):
    """Extract top contributing features for one deal."""
    if isinstance(shap_vals, list):
        sv = shap_vals[1][row_idx]  # Class 1 (win)
    else:
        sv = shap_vals[row_idx]

    # Build (feature, value, shap) tuples
    factors = []
    for i, feat in enumerate(feature_names):
        factors.append({
            "feature": feat,
            "value": X_row.iloc[i] if hasattr(X_row, "iloc") else X_row[i],
            "shap": sv[i],
            "impact": "↑" if sv[i] > 0 else "↓"
        })

    # Sort by absolute SHAP
    factors = sorted(factors, key=lambda x: abs(x["shap"]), reverse=True)

    return factors[:top_n]


def main():
    # Load model & open deals
    model, features, categoricals = load_model_and_features()

    predictions_csv = "results/open_pipeline_predictions.csv"
    score_parquet = "dataset/score.parquet"

    if not Path(score_parquet).exists():
        print(f"ERROR: {score_parquet} not found. Run: python build_features.py")
        return

    # Load open deals
    print(f"Loading features from {score_parquet}...")
    openp = pd.read_parquet(score_parquet)
    X = openp[features].copy()

    # Ensure categorical types
    for c in categoricals:
        X[c] = X[c].astype(str)

    # Get predictions (needed for output)
    if Path(predictions_csv).exists():
        preds = pd.read_csv(predictions_csv)
    else:
        print(f"Warning: {predictions_csv} not found. Run: python predict.py first")
        preds = None

    # Compute SHAP values
    shap_vals, base_val = get_shap_values(model, X)

    # Extract top factors for each deal
    print("Extracting top factors per deal...")
    explanations = []

    for idx in range(len(X)):
        deal_id = openp.iloc[idx].get("display_id", f"deal_{idx}")

        if preds is not None:
            row_pred = preds.iloc[idx]
            prob = row_pred.get("win_probability", 0)
            signal = row_pred.get("signal_strength", "none")
            acv = row_pred.get("acv", 0)
        else:
            prob, signal, acv = 0, "none", 0

        top_factors = extract_top_factors(shap_vals, idx, features, X.iloc[idx], top_n=5)

        explanations.append({
            "display_id": deal_id,
            "win_probability": prob,
            "signal_strength": signal,
            "acv": acv,
            "top_factors_json": str(top_factors),  # JSON-serializable format
            "base_shap_value": base_val if isinstance(base_val, (int, float)) else base_val[1],
            "top_factor_1": top_factors[0]["feature"] if top_factors else "",
            "top_factor_1_impact": top_factors[0]["shap"] if top_factors else 0,
            "top_factor_2": top_factors[1]["feature"] if len(top_factors) > 1 else "",
            "top_factor_2_impact": top_factors[1]["shap"] if len(top_factors) > 1 else 0,
            "top_factor_3": top_factors[2]["feature"] if len(top_factors) > 2 else "",
            "top_factor_3_impact": top_factors[2]["shap"] if len(top_factors) > 2 else 0,
        })

    # Save explanations
    df_expl = pd.DataFrame(explanations)
    output_path = "models/shap_explanations.csv"
    df_expl.to_csv(output_path, index=False)
    print(f"✓ Saved SHAP explanations to {output_path}")

    # Show samples
    print("\n=== Top 5 Deals by Win Probability ===\n")
    if preds is not None:
        top_5 = preds.nlargest(5, "win_probability")[["display_id", "win_probability", "signal_strength"]]
        for _, row in top_5.iterrows():
            deal_id = row["display_id"]
            expl = df_expl[df_expl["display_id"] == deal_id].iloc[0]
            print(f"Deal {deal_id} ({row['win_probability']:.1%}):")
            print(f"  Top factor: {expl['top_factor_1']} (SHAP: {expl['top_factor_1_impact']:.3f})")
            print(f"  Also: {expl['top_factor_2']}, {expl['top_factor_3']}")
            print()


if __name__ == "__main__":
    main()
