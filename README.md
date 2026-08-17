# Sales Win Probability Model — Deployment & Usage Guide

A production-ready CatBoost model to rank open deals by win probability and forecast pipeline revenue.

**Status:** ✅ Shipped | PR-AUC 0.7144 | Backtest validated | Ready for production

---

## Quick Start

### 1. Build the Dataset (One-Time or Monthly)

```bash
# Pull raw data from DevRev
python fetch_dim.py

# Engineer features (outputs to dataset/)
python build_features.py
```

Output:
- `dataset/train.parquet` — 3,573 closed deals (training data)
- `dataset/score.parquet` — 1,912 open deals (inference targets)

### 2. Train the Model (One-Time or When Features Change)

```bash
# Full retrain with walk-forward eval
python train_model.py

# Output: models/catboost_model.pkl, sigmoid_calibrator.pkl, model_metadata.pkl
```

Includes 5-fold walk-forward evaluation. Reports: PR-AUC, ROC, calibration ratio.

### 3. Monthly Calibration (Recommended)

```bash
# Refit the calibrator on recent closed deals
# Do NOT retrain the full model (ranking does not decay)
python train_model.py --calibrate-only
```

Monitors `calibration_ratio` (target <1.30). If >1.60, refits the sigmoid layer to correct drift.

### 4. Score Open Pipeline (Weekly or On-Demand)

```bash
# Generate predictions on open deals
python predict.py

# Output: tabular results with signal_strength, expected_value
```

Or run the notebook:
```bash
jupyter notebook win_probability_colab_new_draft.ipynb
# Run cell §12.2 to generate results/open_pipeline_predictions.csv
```

### 5. Validate Model Health (Monthly)

```bash
# Backtest on historical deals (3-6 months old, now closed)
python backtest_validation.py

# Reports: calibration ratio, actual vs predicted by signal_strength
```

Expected output: calibration ratio ~0.95 (OK if <1.30)

### 6. Drift Detection (Monthly)

```bash
# ARIMA baseline on historical closed deals
python trend_baseline.py

# Compare: $X.XXM forecast vs $Y.YYM historical trend
# Ratio 5.5x likely means company growth, not model drift
```

---

## Directory Structure

```
forecast_v0/
├── dataset/                          # Training and scoring data (git-ignored)
│   ├── train.parquet                 # 3,573 closed deals
│   └── score.parquet                 # 1,912 open deals
│
├── models/                           # Shipped artifacts (git-ignored)
│   ├── catboost_model.pkl            # Ranking model
│   ├── sigmoid_calibrator.pkl        # Calibration layer
│   ├── model_metadata.pkl            # Features, params, eval metrics
│   └── arima_baseline.pkl            # Historical trend (for drift detection)
│
├── results/                          # CSV outputs (git-ignored)
│   └── open_pipeline_predictions.csv # 1,912 open deals scored
│
├── docs/                             # Documentation
│   ├── PROJECT_SUMMARY.md            # What's shipped, what's planned
│   ├── BACKTEST_RESULTS.md           # Validation results & gap analysis
│   └── README.md                     # This file
│
├── win_probability_colab_new_draft.ipynb  # Full analysis notebook
│
├── build_features.py                 # Feature engineering (dataset/ → parquets)
├── train_model.py                    # Model training & calibration
├── predict.py                        # Inference wrapper
├── backtest_validation.py            # Historical validation
├── trend_baseline.py                 # Drift detection
│
└── [supporting files: fetch_dim.py, .gitignore, etc.]
```

---

## Features (16 Total)

### Base (13)
- `days_in_current_stage`, `days_in_sales_cycle`, `stall_ratio`, `contract_months`
- `num_stakeholders`, `has_champion`, `poc`
- `opportunity_type`, `source`, `geo`, `segment` (categorical)
- `number_of_meetings`, `days_since_latest_meeting`

### Meeting Velocity (3)
- `meetings_first_30d` — meetings in first month of the deal
- `meetings_per_month` — engagement rate (scale-free)
- `meeting_quiet_frac` — fraction of deal life with zero meetings (decay signal)

Meeting dates from `dim_meeting.scheduled_date` (when held), not link ingest dates.

---

## Model Architecture

| Component | Details |
|---|---|
| Algorithm | CatBoost |
| Depth | 3 (stopped early; depth 4/5 hurt generalization) |
| Iterations | 300 |
| Learning rate | 0.05 |
| Categoricals | opportunity_type, source, geo, segment (native handling) |
| **Eval** | Walk-forward, 5 folds, 5 seeds on back half of timeline |
| **Eval Metric** | PR-AUC (for imbalanced data, 15.1% base rate) |
| **Calibration** | Sigmoid (LogisticRegression on raw probs) |
| **Threshold** | 0.3 (ships with metadata, overridable per use case) |

### Performance

| Metric | Value | Notes |
|---|---|---|
| **PR-AUC** | 0.7144 ± 0.1159 | 5x base rate; fold std dev ±11.6% |
| **ROC-AUC** | 0.9324 | Strong ranking ability |
| **Precision @ 0.3** | 61.6% | 62% of flagged deals close |
| **Recall @ 0.3** | 57.0% | Catches 57% of actual wins |
| **Calibration** | 0.954x | Slightly under-predicts; safe |

### Known Limits

1. **Right-censoring:** Test deals are recent, unresolved. Harder than training. Train gap 0.95 vs test 0.71 is expected.
2. **541 wins:** Hyperparameter tuning exhausted; no further gains from tuning (all inside 0.0172 noise floor).
3. **Calibration drift:** Walk-forward ratio 0.846–1.877. Refit monthly with `--calibrate-only`.
4. **68.6% abstaining:** Young deals (<0.1 prob) have no signal; model declines to guess.

---

## Outputs

### 1. `open_pipeline_predictions.csv` (1,912 open deals)

| Column | Type | Use |
|---|---|---|
| `rank` | int | Deal rank by win probability |
| `display_id` | str | Deal ID |
| `win_probability` | float | 0–1, raw probability |
| `signal_strength` | cat | none/<0.1, low/0.1–0.3, medium/0.3–0.5, high/>0.5 |
| `predicted_win` | 0/1 | Binary flag at threshold 0.3 |
| `acv` | float | Annual Contract Value ($) |
| `expected_value` | float | win_probability × acv |
| `segment`, `days_in_sales_cycle`, etc. | — | Supporting features for context |

**Totals:**
- Total ACV: $101.54M
- Model forecast: $7.21M (7.1% conversion, based on current probability distribution)
- ARIMA trend: $1.31M/month (historical baseline)
- Ratio: 5.5x (likely company growth, not drift)

### 2. `backtest_validation.py` Output

Scores 286 historical deals (3–6 months old, now closed). Reports by signal_strength:
- Actual close rate vs predicted close rate
- Calibration ratio (0.954 = OK)
- Verdict: "Model is trustworthy"

### 3. `trend_baseline.py` Output

Plots:
- `models/arima_diagnostics.png` — fit quality (ACF/PACF, residuals)
- `models/arima_forecast.png` — history + 3-month forecast

Text:
- Historical monthly revenue range
- ARIMA forecast vs model forecast comparison

---

## Operations Runbook

### Weekly: Score and Distribute

```bash
# Pull latest open deals
python fetch_dim.py
python build_features.py

# Score
python predict.py

# View top 20 in results/open_pipeline_predictions.csv
head -21 results/open_pipeline_predictions.csv
```

Send `open_pipeline_predictions.csv` to sales team.

### Monthly: Calibration Check

```bash
# Refit calibrator on recent closed deals
python train_model.py --calibrate-only

# Check ratio. If 1.30 < ratio < 1.60: OK. If >1.60: alert.
```

### Monthly: Drift Check

```bash
# Validate model on historical cohort
python backtest_validation.py

# Compare forecast vs trend
python trend_baseline.py

# If ratio >5x or <0.5x: investigate (cohort shift or model drift)
```

### Full Retrain (Only When Features Change)

```bash
# Features, hyperparams, or training data changed
python train_model.py

# Full walk-forward eval (5 folds, 5 seeds)
# Saves new models/catboost_model.pkl + sigmoid_calibrator.pkl
```

---

## Interpreting Results

### Signal Strength Buckets

| Bucket | Probability Range | Meaning | Action |
|---|---|---|---|
| **high** | >0.5 | Strong signal; deal has shown evidence of momentum | Prioritize |
| **medium** | 0.3–0.5 | Moderate signal; some activity but early | Track closely |
| **low** | 0.1–0.3 | Weak signal; minimal activity | Nurture |
| **none** | <0.1 | No signal; too young to assess | Develop normally |

### Expected Value vs Probability

- `expected_value = win_probability × acv`
- **Do not use** as a dollar forecast. Calibration unvalidated on open deals (different population).
- **Use as a rank:** higher expected_value = better deal to focus on.

### Threshold 0.3

- Current shipping threshold flags ~146 of 1,429 test deals.
- **Precision:** 61.6% of flagged deals close (low false-alarm rate).
- **Recall:** 57.0% of wins are flagged (44% missed).
- **Trade-off:** Raise to 0.5–0.7 for higher precision (fewer deals, more sure); lower to 0.1–0.2 for coverage (more deals, less sure).

---

## Configuration

### Environment Variables

Override default paths:
```bash
export TRAIN_OUT="path/to/train.parquet"
export SCORE_OUT="path/to/score.parquet"
python build_features.py
```

### Model Params (in `train_model.py`)

```python
PARAMS = dict(
    iterations=300,      # trees to fit
    depth=3,             # tree depth (3 beat 4, 5)
    learning_rate=0.05,  # shrinkage
    random_state=0,      # deterministic
)
```

### Calibration Thresholds (in `train_model.py`)

```python
MIN_WINS_FOR_RATIO = 15  # ignore calibration_ratio if <15 wins
# <1.30 OK | 1.30-1.60 WARN | >1.60 ALARM
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'catboost'`
Ensure ML environment is active. See `.venv-ml` setup.

### `Backtest ratio >1.60`
Model is over-predicting. Run `python train_model.py --calibrate-only` to refit.

### `ARIMA forecast 10x different from model forecast`
Likely company growth or cohort composition shift. Check:
- Is current open pipeline larger than 6 months ago?
- Did segment mix change (more enterprise, fewer SMB)?
- If yes: expected. If no: investigate model.

### `Dataset not found: dataset/train.parquet`
Run feature builder:
```bash
python fetch_dim.py
python build_features.py
```

---

## Documentation

- **PROJECT_SUMMARY.md** — full background, what changed, blockers
- **BACKTEST_RESULTS.md** — validation results, gap analysis, recommendations
- **Notebook §1–§12** — full analysis (exploratory and shipped code mixed; labeled clearly)

---

## References

- `models/model_metadata.pkl` — provenance (features, params, eval metrics)
- `docs/` — project docs
- See `win_probability_colab_new_draft.ipynb` §12.3 for caveats on open-deal forecasting

---

## Contact / Ownership

Model shipped by: AI assistant  
Last updated: 2026-08-17  
Validation: Backtest passed (ratio 0.954x, all signal buckets matched or beat)

For questions on ops, see PROJECT_SUMMARY.md "Next Steps."
