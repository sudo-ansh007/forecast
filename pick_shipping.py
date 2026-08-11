"""Run the top-4 candidates through the FULL shipping path and compare what ships.

Test-set PR-AUC is not the deliverable -- the deliverable is a calibrated probability
that gets multiplied by ACV. So this scores each candidate on the time split AND runs it
through make_dataset.py's actual pipeline (same encode, same calibrator, same evidence
gate) to see what each one does to the committed forecast.

Usage:  python pick_shipping.py
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from make_dataset import FEATURES, LABEL, MIN_EVIDENCE, encode

SEEDS = range(5)


def cal(m):
    """The shipping calibration wrapper, identical to make_dataset.main()."""
    return CalibratedClassifierCV(m, method="sigmoid",
                                  cv=StratifiedKFold(3, shuffle=True, random_state=0))


# The four that beat or matched shipping on the new dataset, plus shipping itself.
# CatBoost raw is UNCALIBRATED on purpose -- that is the configuration that scored
# 0.630, and wrapping it changes the thing being tested. Both forms are run.
CANDIDATES = {
    "LightGBM SHIPPING": lambda s: cal(LGBMClassifier(
        max_depth=3, num_leaves=8, min_child_samples=50, n_estimators=200,
        learning_rate=0.05, random_state=s, verbose=-1, n_jobs=-1)),
    "XGBoost min_child_wt1": lambda s: cal(XGBClassifier(
        max_depth=3, n_estimators=200, learning_rate=0.05, min_child_weight=1,
        random_state=s, n_jobs=-1, eval_metric="logloss")),
    "CatBoost d3": lambda s: cal(CatBoostClassifier(
        depth=3, iterations=200, learning_rate=0.05, min_data_in_leaf=50,
        random_seed=s, verbose=0, allow_writing_files=False, thread_count=-1)),
    "CatBoost d3 it600 lr.03": lambda s: cal(CatBoostClassifier(
        depth=3, iterations=600, learning_rate=0.03, min_data_in_leaf=50,
        random_seed=s, verbose=0, allow_writing_files=False, thread_count=-1)),
    "CatBoost raw (uncal)": lambda s: CatBoostClassifier(
        depth=3, iterations=200, learning_rate=0.05, min_data_in_leaf=50,
        random_seed=s, verbose=0, allow_writing_files=False, thread_count=-1),
    "CatBoost raw + sigmoid": lambda s: cal(CatBoostClassifier(
        depth=3, iterations=200, learning_rate=0.05, min_data_in_leaf=50,
        random_seed=s, verbose=0, allow_writing_files=False, thread_count=-1)),
}


def ece(y, p, bins=10):
    """Expected calibration error -- mean |predicted - actual| weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return tot


def recall_at(y, p, frac):
    k = max(1, int(len(p) * frac))
    return y[np.argsort(p)[::-1][:k]].sum() / y.sum()


def main():
    train = pd.read_parquet("train.parquet").sort_values("created_date").reset_index(drop=True)
    score = pd.read_parquet("score.parquet")

    X, X_score = encode(train, score)
    y = train[LABEL].to_numpy()
    cut = int(len(train) * 0.8)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    print(f"train {cut:,}/{ytr.sum()} won   test {len(yte):,}/{yte.sum()} won "
          f"(base {yte.mean():.1%})\n")

    ev = score["evidence"].to_numpy()
    gated = ev >= MIN_EVIDENCE
    acv = score["acv"].to_numpy()

    rows = []
    for name, make in CANDIDATES.items():
        pr, br, ec, rc, r20, ratio = [], [], [], [], [], []
        for s in SEEDS:
            p = make(s).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            pr.append(average_precision_score(yte, p))
            br.append(brier_score_loss(yte, p))
            ec.append(ece(yte, p))
            rc.append(roc_auc_score(yte, p))
            r20.append(recall_at(yte, p, 0.2))
            ratio.append(p.sum() / yte.sum())

        # Refit on ALL of train and score the open pipeline, exactly as main() does.
        p_open = make(0).fit(X, y).predict_proba(X_score)[:, 1]
        rows.append({
            "model": name,
            "PR_AUC": np.mean(pr), "sd": np.std(pr),
            "ROC": np.mean(rc), "Brier": np.mean(br), "ECE": np.mean(ec),
            "rec@20": np.mean(r20), "sum_p_over_act": np.mean(ratio),
            "committed_wins": p_open[gated].sum(),
            "committed_$M": (p_open[gated] * acv[gated]).sum() / 1e6,
            "median_p": np.median(p_open[gated]),
            "max_p": p_open[gated].max(),
        })
        print(f"  done {name}")

    df = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    pd.set_option("display.width", 200)
    print("\n=== TEST SET (5 seeds) ===")
    print(df[["model", "PR_AUC", "sd", "ROC", "Brier", "ECE", "rec@20",
              "sum_p_over_act"]].round(4).to_string(index=False))
    print("\n=== WHAT EACH ONE SHIPS (622 gated open deals) ===")
    print(df[["model", "committed_wins", "committed_$M", "median_p",
              "max_p"]].round(3).to_string(index=False))

    best = df.iloc[0]
    ship = df[df.model == "LightGBM SHIPPING"].iloc[0]
    print(f"\ntop PR-AUC: {best.model} {best.PR_AUC:.4f} vs shipping {ship.PR_AUC:.4f} "
          f"(gap {best.PR_AUC - ship.PR_AUC:+.4f}, shipping sd {ship.sd:.4f})")
    print(f"calibration: best {best.sum_p_over_act:.2f}x vs shipping "
          f"{ship.sum_p_over_act:.2f}x -- forecast differs by "
          f"${abs(best['committed_$M'] - ship['committed_$M']):.2f}M")
    df.to_csv("shipping_comparison.csv", index=False)
    print("\nwrote shipping_comparison.csv")


if __name__ == "__main__":
    main()
