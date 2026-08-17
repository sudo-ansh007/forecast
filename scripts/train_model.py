#!/usr/bin/env python3
"""Train the win-probability model and save artifacts.

    python train_model.py                 # full retrain + walk-forward eval
    python train_model.py --calibrate-only  # refit the calibrator, keep the model

WHY TWO MODES: ranking does not decay, calibration does. A model frozen 14
months measured a refit advantage of +0.0001 PR_AUC (2 of 5 folds) -- inside
seed noise -- while a frozen calibrator drifted to 1.17-2.29x over-prediction.
So the recurring job is --calibrate-only; a full retrain is only needed when
features change. See calibration_ratio() for the trigger.

REPORTED EVAL IS WALK-FORWARD, not a single chronological slice. The old single
20% slice scored PR_AUC 0.582 with a +/-0.10 CI because it happened to be the
most right-censored window in the timeline (only 79 wins). Five sequential folds
over the back half of the timeline put the same model at 0.71.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATASET_DIR = Path("dataset")
TRAIN_PATH = DATASET_DIR / "train.parquet"
OUT_DIR = Path("models")

# 13 base + 3 meeting-velocity. The velocity columns are scale-free (per-month,
# or a fraction of deal life) so they do not drift with deal age the way a raw
# all-time meeting count does.
FEATURES = [
    "days_in_current_stage", "days_in_sales_cycle", "stall_ratio", "contract_months",
    "num_stakeholders", "has_champion", "poc", "opportunity_type", "source", "geo",
    "segment", "number_of_meetings", "days_since_latest_meeting",
    "meetings_first_30d", "meetings_per_month", "meeting_quiet_frac",
]
CATEGORICALS = ["opportunity_type", "source", "geo", "segment"]

# depth 3 beat 4 and 5 on walk-forward (0.701 / 0.692 / 0.688 PR_AUC). Note
# make_dataset.py also uses depth 3 but one-hots the categoricals instead of
# passing cat_features -- that is a separate reference path, not this one.
PARAMS = dict(iterations=300, depth=3, learning_rate=0.05, random_state=0, verbose=False)

CAL_FRAC = 0.15   # trailing share of the timeline reserved for the calibrator
N_FOLDS = 5
MIN_WINS_FOR_RATIO = 15


def load():
    df = pd.read_parquet(TRAIN_PATH)
    df["created_date"] = pd.to_datetime(df["created_date"])
    return df.sort_values("created_date").reset_index(drop=True)


def as_X(df):
    X = df[FEATURES].copy()
    for c in CATEGORICALS:
        X[c] = X[c].astype(str)
    return X


def fit(X, y):
    cat_idx = [X.columns.get_loc(c) for c in CATEGORICALS]
    return CatBoostClassifier(cat_features=cat_idx, **PARAMS).fit(X, y)


def calibrate(model, X_cal, y_cal):
    raw = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    return LogisticRegression().fit(raw, y_cal)


def predict(model, cal, X):
    raw = model.predict_proba(X)[:, 1].reshape(-1, 1)
    return cal.predict_proba(raw)[:, 1]


def calibration_ratio(p, y_true):
    """Ops monitor: sum(predicted) / actual wins. Run monthly on deals resolved
    since the last check. This is the number to watch -- not PR_AUC.

    <1.30 OK | 1.30-1.60 refit the calibrator | >1.60 refit now
    Under 15 wins the ratio is noise; do not act on it.
    """
    if y_true.sum() < MIN_WINS_FOR_RATIO:
        return "INSUFFICIENT", float("nan")
    ratio = p.sum() / y_true.sum()
    state = "OK" if ratio < 1.30 else ("WARN refit calibrator" if ratio < 1.60 else "ALARM refit now")
    return state, float(ratio)


def walk_forward(df):
    """Five sequential chronological folds over the back half of the timeline.
    Each fold trains on prior rows only and calibrates on the slice immediately
    before its test range, so no fold ever sees its own future.
    """
    X, y, n = as_X(df), df["is_won"].to_numpy(), len(df)
    qs = np.linspace(0.50, 0.90, N_FOLDS + 1)
    rows = []
    for i in range(N_FOLDS):
        tr_hi = int(n * (qs[i] - CAL_FRAC))
        cal_hi, te_hi = int(n * qs[i]), int(n * qs[i + 1])
        model = fit(X[:tr_hi], y[:tr_hi])
        cal = calibrate(model, X[tr_hi:cal_hi], y[tr_hi:cal_hi])
        p, yy = predict(model, cal, X[cal_hi:te_hi]), y[cal_hi:te_hi]
        rows.append(dict(
            fold=i + 1, n_train=tr_hi, n_test=len(yy), wins=int(yy.sum()),
            PR_AUC=average_precision_score(yy, p), ROC=roc_auc_score(yy, p),
            Brier=brier_score_loss(yy, p),
            P_at_20=yy[np.argsort(p)[::-1][:20]].mean(),
            cal_ratio=p.sum() / yy.sum(),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate-only", action="store_true",
                    help="refit the calibrator against recent labels, keep the saved model")
    args = ap.parse_args()

    df = load()
    X, y = as_X(df), df["is_won"].to_numpy()
    cal_lo = int(len(df) * (1 - CAL_FRAC))
    OUT_DIR.mkdir(exist_ok=True)

    if args.calibrate_only:
        with open(OUT_DIR / "catboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        old = pickle.load(open(OUT_DIR / "sigmoid_calibrator.pkl", "rb"))
        recent_X, recent_y = X[cal_lo:], y[cal_lo:]
        before = calibration_ratio(predict(model, old, recent_X), recent_y)
        cal = calibrate(model, recent_X, recent_y)
        after = calibration_ratio(predict(model, cal, recent_X), recent_y)
        print(f"recalibrated on {len(recent_y)} recent deals ({recent_y.sum()} wins)")
        print(f"  ratio before {before[1]:.3f} ({before[0]})  ->  after {after[1]:.3f} ({after[0]})")
        with open(OUT_DIR / "sigmoid_calibrator.pkl", "wb") as f:
            pickle.dump(cal, f)
        print(f"Saved {OUT_DIR}/sigmoid_calibrator.pkl (model untouched)")
        return

    print(f"{len(df)} closed deals, {y.sum()} won (base rate {y.mean():.3f})")

    wf = walk_forward(df)
    print("\nWalk-forward eval (reported):")
    print(wf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  PR_AUC {wf.PR_AUC.mean():.4f} (fold SD {wf.PR_AUC.std():.4f})"
          f"   ROC {wf.ROC.mean():.4f}   P@20 {wf.P_at_20.mean():.4f}")
    print(f"  calibration ratio across folds: {wf.cal_ratio.min():.3f} .. {wf.cal_ratio.max():.3f}")

    # Ship: model on everything up to the calibration slice, calibrator on that slice.
    model = fit(X[:cal_lo], y[:cal_lo])
    cal = calibrate(model, X[cal_lo:], y[cal_lo:])
    print(f"\nfinal fit: {cal_lo} train rows, {len(df) - cal_lo} calibration rows")

    meta = dict(
        features=FEATURES, categoricals=CATEGORICALS, params=PARAMS,
        eval="walk-forward, %d folds over the back half of the timeline" % N_FOLDS,
        wf_pr_auc=wf.PR_AUC.mean(), wf_pr_auc_fold_sd=wf.PR_AUC.std(),
        wf_roc=wf.ROC.mean(), wf_p_at_20=wf.P_at_20.mean(),
        wf_cal_ratio_range=(wf.cal_ratio.min(), wf.cal_ratio.max()),
        threshold=0.3, train_rows=cal_lo, cal_rows=len(df) - cal_lo,
        meeting_date_source="dim_meeting.scheduled_date (held date, not link ingest)",
    )
    for name, obj in (("catboost_model", model), ("sigmoid_calibrator", cal),
                      ("model_metadata", meta)):
        with open(OUT_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    print(f"Saved 3 artifacts to {OUT_DIR}/")
    print("\nRecurring job: python train_model.py --calibrate-only  (monthly, or when "
          "calibration_ratio hits WARN). Full retrain only when features change.")


if __name__ == "__main__":
    main()
