#!/usr/bin/env python3
"""Inference wrapper for win probability model."""
import pickle
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("dataset")

class WinProbabilityModel:
    def __init__(self, model_dir="models"):
        """Load trained model + calibrator."""
        model_dir = Path(model_dir)

        with open(model_dir / "catboost_model.pkl", "rb") as f:
            self.catboost_model = pickle.load(f)

        with open(model_dir / "sigmoid_calibrator.pkl", "rb") as f:
            self.sigmoid_cal = pickle.load(f)

        with open(model_dir / "model_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.features = self.metadata["features"]
        self.categoricals = self.metadata["categoricals"]
        self.threshold = self.metadata["threshold"]

    def predict_proba(self, df):
        """Return calibrated win probabilities.

        Args:
            df: DataFrame with feature columns

        Returns:
            array of win probabilities (0-1)
        """
        # Prepare features
        X = df[self.features].copy()
        for c in self.categoricals:
            X[c] = X[c].astype(str)

        # Raw CatBoost predictions
        p_raw = self.catboost_model.predict_proba(X)[:, 1]

        # Sigmoid calibration
        p_cal = self.sigmoid_cal.predict_proba(p_raw.reshape(-1, 1))[:, 1]

        return p_cal

    def predict(self, df):
        """Return binary predictions (win/lose) at default threshold.

        Args:
            df: DataFrame with feature columns

        Returns:
            array of 0/1 predictions
        """
        p = self.predict_proba(df)
        return (p >= self.threshold).astype(int)

    def predict_with_confidence(self, df):
        """Return predictions + probabilities + confidence bucket.

        Returns:
            DataFrame with columns: probability, prediction, confidence_bucket
        """
        p = self.predict_proba(df)
        pred = (p >= self.threshold).astype(int)

        # ponytail: simple bucketing, no fancy thresholds
        buckets = pd.cut(p, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=["very_low", "low", "medium", "high", "very_high"])

        return pd.DataFrame({
            "probability": p,
            "prediction": pred,
            "confidence_bucket": buckets
        })


if __name__ == "__main__":
    # Demo
    model = WinProbabilityModel()

    # dataset/train.parquet, not ml_train.csv -- the CSV predates the meeting-velocity
    # columns and its meeting timing came from link ingest dates.
    train = pd.read_parquet(DATASET_DIR / "train.parquet").sort_values("created_date").reset_index(drop=True)
    test = train.iloc[int(len(train) * 0.8):].reset_index(drop=True)

    # Predict
    results = model.predict_with_confidence(test)
    results["actual"] = test.is_won.values

    print("Sample predictions:")
    print(results.head(10).to_string(index=False))

    print(f"\nAccuracy at threshold {model.threshold}: {(results.prediction == results.actual).mean():.3f}")
    print(f"Precision: {(results[results.prediction == 1].actual).mean():.3f}")
    print(f"Recall: {results[results.actual == 1].prediction.mean():.3f}")
