"""Evaluation harness for deal win-probability models.

Scores every candidate model identically so comparison is apples-to-apples:
  - PR-AUC          primary ranking metric (baseline = base rate, ~0.049)
  - Brier           calibration; THE metric, since the product is a revenue number
  - ROC-AUC         reported, not optimised (inflated by 10.5k true negatives)
  - Recall@top20%   operational reading for a sales manager
  - Reliability     10-bin predicted-vs-actual table

Baselines that must be beaten:
  1. stage_constant  -- the incumbent forecast (amount x per-stage win rate). Ship gate.
  2. stage_only      -- logistic regression on stage_num alone. Guards against a
                        26-feature model that secretly learned only stage.

Split is by created_date (oldest 80% train / newest 20% test), never random:
rep_win_rate and account history leak future outcomes into the past otherwise.

Usage:
    python build_features.py && python evaluate.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_features import (CATEGORICALS, CONSERVATIVE_FEATURES, LEAK_SAFE_FEATURES,
                            NO_AMOUNT_FEATURES)

TEST_FRACTION = 0.2
RELIABILITY_BINS = 10
TOP_K_FRACTION = 0.2
RANDOM_STATE = 0


def encode(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """One-hot the categoricals, aligning test columns to train's."""
    num = [f for f in features if f not in CATEGORICALS]
    cats = [f for f in features if f in CATEGORICALS]

    def build(df):
        out = df[num].astype(float).copy()
        for c in cats:
            out = pd.concat([out, pd.get_dummies(df[c], prefix=c, dtype=float)], axis=1)
        return out

    xtr, xte = build(train), build(test)
    # a category present only in test would otherwise shift column order
    xte = xte.reindex(columns=xtr.columns, fill_value=0.0)
    return xtr.to_numpy(), xte.to_numpy(), list(xtr.columns)


def recall_at_top_k(y, p, frac=TOP_K_FRACTION):
    """Of deals that actually won, what share landed in the top-scoring frac?"""
    k = max(1, int(len(p) * frac))
    top = np.argsort(p)[::-1][:k]
    return y[top].sum() / y.sum() if y.sum() else np.nan


def reliability(y, p, bins=RELIABILITY_BINS):
    """Predicted vs actual win rate per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
            "n": int(m.sum()),
            "predicted": p[m].mean(),
            "actual": y[m].mean(),
            "gap": p[m].mean() - y[m].mean(),
        })
    return pd.DataFrame(rows)


def score(name, y, p):
    return {
        "model": name,
        "PR-AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "Recall@20%": recall_at_top_k(y, p),
        "mean_pred": p.mean(),
    }


def stage_constant_baseline(train, test):
    """The incumbent forecast: per-stage historical win rate, applied as a constant.

    This is what the deterministic forecast does today (admin-set constant x amount),
    reproduced empirically. Beating it is the ship/no-ship gate.
    """
    rates = train.groupby("stage_num")["is_won"].mean()
    return test["stage_num"].map(rates).fillna(train["is_won"].mean()).to_numpy()


def main():
    df = pd.read_parquet("train.parquet")
    df = df.sort_values("created_date").reset_index(drop=True)

    cut = int(len(df) * (1 - TEST_FRACTION))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    ytr, yte = train["is_won"].to_numpy(), test["is_won"].to_numpy()

    print(f"train {len(train)} deals / {ytr.sum()} won ({ytr.mean():.3f})")
    print(f"test  {len(test)} deals / {yte.sum()} won ({yte.mean():.3f})")
    if yte.sum() < 30:
        print(f"WARNING: only {yte.sum()} test positives -- metrics are noisy, "
              "read the CV spread not the point estimate")

    results = []
    results.append(score("base_rate (always predict mean)",
                         yte, np.full(len(yte), ytr.mean())))

    # NOTE: the stage_constant and stage_only baselines are omitted on purpose.
    # On closed deals stage_num IS the label (8=won, 9=lost), so both score a
    # meaningless 1.0. They only become measurable under point-in-time
    # reconstruction, where stage is the value as of the observation date.
    print("\nSKIPPED baselines: stage_constant, stage_only "
          "(stage_num == label on closed deals; needs point-in-time to evaluate)")

    p_cal = None
    for label, feats in [("full", LEAK_SAFE_FEATURES),
                         ("conservative", CONSERVATIVE_FEATURES),
                         ("no_amount", NO_AMOUNT_FEATURES)]:
        xtr, xte, names = encode(train, test, feats)

        # C=0.1 regularises hard: 541 positives across ~17 features overfits at defaults.
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        lr.fit(xtr, ytr)
        results.append(score(f"logreg [{label}]", yte, lr.predict_proba(xte)[:, 1]))

        def make_hgb():
            # Depth 3 + min_samples_leaf 50: shallow on purpose at this label count.
            # No class_weight -- it wrecks calibration, which matters more than ranking.
            return HistGradientBoostingClassifier(
                max_depth=3, min_samples_leaf=50, max_iter=200, learning_rate=0.05,
                early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE,
            )

        hgb = make_hgb()
        hgb.fit(xtr, ytr)
        results.append(score(f"hgb [{label}]", yte, hgb.predict_proba(xte)[:, 1]))

        # Gradient boosting is systematically miscalibrated; Platt (sigmoid) rather
        # than isotonic because isotonic needs several hundred positives to behave.
        cal = CalibratedClassifierCV(make_hgb(), method="sigmoid", cv=3)
        cal.fit(xtr, ytr)
        p = cal.predict_proba(xte)[:, 1]
        results.append(score(f"hgb_calibrated [{label}]", yte, p))

        if label == "no_amount":
            p_cal, cons_lr, cons_names = p, lr, names

    out = pd.DataFrame(results).set_index("model")
    print("\n=== METRICS (test = newest 20% by created_date) ===")
    print(out.round(4).to_string())
    print(f"\nPR-AUC baseline (random) = {yte.mean():.4f}")

    print("\n=== RELIABILITY: hgb_calibrated [no_amount] ===")
    print(reliability(yte, p_cal).round(4).to_string(index=False))

    # Aggregate sanity check: does the model imply a plausible revenue number?
    print("\n=== AGGREGATE FORECAST CHECK (test slice) ===")
    exp_deals = p_cal.sum()
    print(f"predicted wins {exp_deals:.1f} vs actual {yte.sum()} "
          f"({exp_deals / max(yte.sum(), 1):.2f}x)")
    amt = test["acv"].to_numpy()
    print(f"predicted revenue ${(p_cal * amt).sum() / 1e6:.1f}M "
          f"vs actual ${(yte * amt).sum() / 1e6:.1f}M")
    print("NOTE: amount is real on only ~41% of open deals -- treat revenue as a range, "
          "not a point, and report the coverage alongside it")

    # Drivers: linear coefficients are the honest read at this label count.
    coefs = pd.Series(cons_lr[-1].coef_[0], index=cons_names).sort_values(key=abs, ascending=False)
    print("\n=== TOP 12 DRIVERS (logistic coefficients, no_amount set) ===")
    print(coefs.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
