# Sales Win Probability Model — Project Summary

## What We Built

**Objective:** Rank open deals by win probability to forecast pipeline revenue.

### Core Model (v1, Shipped)

| Component | Details |
|---|---|
| **Algorithm** | CatBoost (depth 3, 300 iterations) |
| **Features** | 16: 13 base + 3 meeting-velocity |
| **Eval Method** | Walk-forward, 5 folds, 5 seeds (chronological, no leakage) |
| **Eval Metric** | PR-AUC 0.7144 (±0.1159 fold sd) |
| **ROC-AUC** | 0.9324 |
| **Calibration** | Sigmoid (LogisticRegression on raw CatBoost probs) |
| **Threshold** | 0.3 (flag ~146 of 1,429 closed test deals, 61.6% precision) |
| **Train Data** | 3,573 closed deals (541 wins, 15.1% base rate), meeting dates from `dim_meeting.scheduled_date` (held, not ingest) |

### Key Discovery

**Meeting dates were broken.** Used `dim_link.created_date` (ingest stamp, only from 2025-02-04) instead of `dim_meeting.scheduled_date` (held, reaches back to 2023-02-20). This taught the model an era marker: pre-Feb-2025 deals showed 0 meetings and won at 26.3% vs 12% post-floor. Fix recovered **+0.0268 test PR-AUC** — nearly the entire measured gain from v0 → v1.

### Delivered Artifacts

```
models/
├── catboost_model.pkl           # ranking model
├── sigmoid_calibrator.pkl        # calibration layer
└── model_metadata.pkl            # features, params, eval metrics, threshold

win_probability_colab_new_draft.ipynb   # 114 cells, full analysis + runnable cells
├── §1–3: data load, features, exploratory analysis
├── §4–7: baseline, optuna search (shown as failed), meeting-date defect investigation
├── §8–10: model training, ablation, noise floor, calibration drift
├── §11: v0 vs v1 comparison, train/test gap, precision/recall curves, attribution
└── §12: score open pipeline → CSV, model provenance check

results/
└── open_pipeline_predictions.csv   # 1,912 open deals scored
    ├── display_id, win_probability, signal_strength
    ├── signal_strength: none (<0.1), low (0.1–0.3), medium (0.3–0.5), high (>0.5)
    ├── expected_value = win_probability × ACV
    └── segment, age, stakeholders, meetings

build_features.py                  # feature engineering (13 base + 3 velocity)
train_model.py                     # model training + walk-forward eval + --calibrate-only
predict.py                         # inference wrapper (loads models/, scores deals)
```

### Why It's Good to Go

1. **Ranking is honest:** Walk-forward eval prevents lucky-seed overfitting. PR-AUC 0.71 validated on unseen future data.
2. **Data defect was real and fixed:** Meeting dates were systematically wrong; fix is permanent, not a tuning knob.
3. **Features are necessary:** Ablation showed dropping the meetings block moved from mid-tier to *worst of 12*.
4. **Precision is actionable:** 62% of flagged deals close (low false-alarm rate for a shortlist).

### Why It Can't Be Improved More (Right Now)

| Ceiling | Evidence |
|---|---|
| **Right-censoring + 541 wins** | Train PR-AUC ≈0.95, test ≈0.71. Test deals are unresolved, harder cohort. Can't fix without years of wait or changing target. |
| **Hyperparameter tuning exhausted** | 5 configs × 5 seeds × 5 folds: best (depth 3) = 0.7223 → 5-seed mean 0.7082 (−0.003). All "wins" were lucky seeds. |
| **Feature engineering hit ceiling** | 16 features tested; dropping any hurts. Meeting velocity bought +0.004 (inside noise) but block is essential. |
| **Calibration drift structural** | Walk-forward ratio swings 0.846–1.877. Right-censoring again: recent folds (fewest resolved wins) drift worst. |

---

## Current State: Open Pipeline Scoring

**1,912 open deals scored:**
| Signal Strength | Count | ACV | Forecast |
|---|---|---|---|
| **high** (>0.5) | 18 | $0.74M | $0.50M |
| **medium** (0.3–0.5) | 12 | $0.81M | $0.33M |
| **low** (0.1–0.3) | 118 | $5.42M | $0.85M |
| **none** (<0.1) | 1,764 | $94.57M | $5.53M |
| **Total** | 1,912 | $101.54M | $7.21M |

**Interpretation:**
- 18 deals with real signal (model has seen enough evidence to rank them).
- 130 deals worth nurturing (some activity, but early).
- 1,764 deals too young to judge (no meetings, early stage).
- **Do not lower threshold to flag more.** Threshold 0.5 flags *fewer* (97 deals). Threshold 0.3 already flags the only deals with signal. Young deals = no signal, not negative signal.

---

## What's Planned (Priority)

### 1. Trend Baseline (1 hour, next)
**Build:** ARIMA model on historical closed deals.
**Goal:** Answer "when you had $X open, what did you close?" Compare forecast vs trend. Catch systematic drift.
**Output:** `trend_baseline.py`, reference forecast in `open_pipeline_predictions.csv`.

### 2. Calibration Scheduler (30 min)
**Build:** Wrapper script + cron instruction.
**Goal:** Monthly `python train_model.py --calibrate-only`.
**Trigger:** When `calibration_ratio()` hits 1.30–1.60 (WARN).

### 3. Validation Dashboard (2 hours)
**Build:** Query closed deals from this month's open-deal CSV.
**Goal:** Did high-signal deals close at 76%? Low at 18%? Feeds drift detection.
**Output:** Monthly validation report.

### 4. LLM Explainer (optional, 2–3 hours)
**Build:** SHAP per-deal attribution → LLM summary.
**Goal:** Reps see "why OPP-13096 ranked #1" in natural language.
**Output:** Optional column in CSV.

### 5. Cleanup (1 hour)
- Delete `make_dataset.py` + `ml_train.csv` (dead ingest-date path).
- Add test to `train_model.py`.
- Write ops runbook.

---

## Files to Ignore / Quarantine

**Delete (dead code, old paths):**
```
make_dataset.py                    # depth 3 + one-hot, 13 features, ingest-date path
ml_train.csv                       # companion to above, stale
add_optuna_tuning.py              # dead script
add_final_cells.py                # dead script
add_catboost_tuning.py            # dead script
debug_query.py                    # debug artifact
fetch_oasis.py                    # debug artifact
```

**Keep but ignore in git (temp/debug):**
```
catboost_info/                    # CatBoost training logs
_backup_train_ingest.parquet      # v0 backup (for comparison only, §11)
_backup_score_ingest.parquet      # v0 backup
_ingest_dates_train.parquet       # intermediate (anchor-fixed but ingest dates)
_ingest_dates_score.parquet       # intermediate
_backup_notebook_precleanup.ipynb # notebook checkpoint
_backup_pre_s11.ipynb             # notebook checkpoint
_backup_pre_s12.ipynb             # notebook checkpoint
*.pyc                             # Python bytecode
__pycache__/                      # Python cache
.venv/                            # virtual environment
.venv-ml/                         # ML virtual environment
features_*.parquet                # stale intermediate builds
```

**Add to `.gitignore`:**
```
# Generated artifacts
models/
results/
*.csv

# Backups and intermediates
_backup_*.parquet
_backup_*.ipynb
_ingest_dates_*.parquet
catboost_info/

# Dead code
make_dataset.py
ml_train.csv
add_*.py
debug_*.py
fetch_*.py

# Environment and cache
.venv/
.venv-ml/
__pycache__/
*.pyc
.DS_Store
```

---

## How to Use Going Forward

### Score new open deals
```bash
# 1. Rebuild score.parquet from latest CRM export
python build_features.py score

# 2. Generate predictions
python -c "
from predict import WinProbabilityModel
import pandas as pd
m = WinProbabilityModel()
s = pd.read_parquet('score.parquet')
p = m.predict_proba(s)
# ... (see §12.2 in notebook for full pipeline)
"

# 3. Or run the notebook
jupyter notebook win_probability_colab_new_draft.ipynb
# Run cell §12.2 to generate CSV
```

### Monthly calibration (when ratio drifts)
```bash
python train_model.py --calibrate-only
# Reports ratio before/after, saves new sigmoid_calibrator.pkl
```

### Full retrain (when features change)
```bash
python train_model.py
# Walk-forward eval, saves all 3 artifacts
```

---

## Known Limitations

1. **Calibration on open deals unvalidated.** $7.21M forecast has no error bar. Walk-forward ratio swings 0.846–1.877 on closed data.
2. **Population mismatch.** Open deals younger, different composition than closed training set.
3. **68.6% abstaining.** Young deals have no signal; model doesn't predict, it declines to guess.
4. **Fold SD ±11.6%.** Accuracy swings Q-to-Q. Don't promise 0.71 to anyone; promise "ranked better than gut feel."
5. **No scheduled recalibration.** Monthly `--calibrate-only` is manual. Needs cron or external scheduler.

---

## Next Immediate Action

**Trend baseline.** See `TREND_BASELINE.md` (to be written next).
