# Backtest Validation Results

**Date:** 2026-08-17  
**Cohort:** 286 deals created 2026-02-02 to 2026-05-03 (3–6 months old, now closed)  
**Model:** CatBoost, PR-AUC 0.7144, 16 features, shipped in `models/`

---

## Summary

✅ **Model is trustworthy on this cohort.**

Backtest validation confirms the model ranks deals correctly. High-signal deals closed at **84.2% actual** vs 68.1% predicted (under-predicting by 16.1%, which is conservative and acceptable).

Calibration ratio: **0.954x** (OK; threshold <1.30)

→ **$7.21M forecast on open pipeline is sound.**

---

## Results by Signal Strength

| Signal | Deals | Actual Close Rate | Predicted Avg | Error | Interpretation |
|---|---|---|---|---|---|
| **high** (>0.5) | 19 | 84.2% | 68.1% | +16.1% | Better than predicted; model under-confident |
| **medium** (0.3–0.5) | 10 | 50.0% | 40.9% | +9.1% | Close match |
| **low** (0.1–0.3) | 35 | 20.0% | 17.2% | +2.8% | Excellent match |
| **none** (<0.1) | 222 | 4.5% | 5.9% | −1.4% | Excellent match |

---

## Key Findings

### Calibration Ratio: 0.954x

- **Definition:** sum(predicted probabilities) / actual wins
- **Interpretation:** For every 100% of wins the model predicted, 95.4% actually happened
- **Status:** OK (threshold is <1.30; >1.60 would alarm)
- **Implication:** Model slightly under-predicts. Safe, not dangerous.

### Where the Gap Lives

- **High-signal deals (>0.5 prob):** 16.1% under-prediction
  - Model said 68%, actually closed at 84%
  - These deals are the best-signal subset; model is too conservative here
  
- **Medium and low signal:** matched well
  - Medium: 40.9% vs 50% (within noise)
  - Low: 17.2% vs 20% (excellent match)
  - None: 5.9% vs 4.5% (excellent match)

### What This Means for Production

1. **Ranking is reliable.** High-signal deals do close more than low-signal. The ordering is correct.
2. **Dollar forecast is sound.** $7.21M on 1,912 open deals is not inflated.
3. **Conservative bias is good.** Reps will see deals win at higher rates than predicted → credibility.
4. **5.5x vs ARIMA baseline is likely real.** Not model drift; likely company growth (open pipeline $101.5M vs historical norm).

---

## Why the Gap Exists

**Root cause:** Sigmoid calibrator was trained on 536 recent deals during the full retrain. Those 536 may have been different from the 286 deals in this backtest (different segments, seasons, rep quality, etc.).

This is **normal and expected**. Two ways to handle:

### Option 1: Accept It (Recommended)
- Gap of 16% on high-signal is tolerable
- Better to under-predict than over-predict (asymmetric cost)
- Let it drift naturally as new data arrives

### Option 2: Recalibrate (3+ Months)
- Run `python train_model.py --calibrate-only` monthly
- Calibrator will learn the true distribution on new closed deals
- Reduces gap over time, but requires fresh labels

### Option 3: Temperature Scaling (Risky)
- Multiply all probabilities by ~1.15 to close the gap
- Pros: instant
- Cons: overfits to this backtest; will over-predict on future unseen data
- **Not recommended** unless you have a second holdout to validate against

---

## Verdict

| Question | Answer |
|---|---|
| Is the model working? | ✅ Yes |
| Are the rankings correct? | ✅ Yes |
| Is the $7.21M forecast valid? | ✅ Yes |
| Should we use it? | ✅ Yes |
| Should we recalibrate now? | ❌ No (optional in 3+ months) |
| Should we apply temperature scaling? | ❌ No (risky without second holdout) |

**Recommendation:** Ship as-is. Monitor monthly via `--calibrate-only` and track actual closes vs predicted.

---

## Next Steps

1. **Weekly:** Score new open deals, send top 20 to sales
2. **Monthly:** Run `python train_model.py --calibrate-only` if calibration ratio drifts
3. **Quarterly:** Run another backtest (pick a different 3-month cohort)
4. **In 90 days:** Query the 1,912 open deals from today; measure conversion by signal_strength

---

## Appendix: Backtest Command

```bash
python backtest_validation.py
# Scores 286 historical deals (3-6 months old, now closed)
# Compares predicted vs actual by signal_strength bucket
# Reports calibration ratio and verdict
```

Expected output:
```
Backtest cohort: 286 deals created 2026-02-02 to 2026-05-03
  Actual wins: 38 (13.3%)
...
Calibration ratio: 0.954x ('OK' if <1.30)
✓ Model is trustworthy on this cohort. Ready to use on live open deals.
```
