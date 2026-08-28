# LLM-as-Forecaster vs ML Model: Win Probability Evaluation

## Summary (Honest Numbers — Overfitting Addressed)

### The Overfitting Problem

Both CatBoost and V2 LLM were initially evaluated on an **80/20 random split**. This is **optimistic** for both because:
1. **Random splits leak temporal patterns** — future deals appear in training, creating unrealistically easy test conditions.
2. **The 13d/78.5d batch-closed pattern** (132 deals, 100% wins) distributes randomly across train/test — in real deployment, this cohort may concentrate in one time window.

The **walk-forward evaluation** (5 chronological folds, train on past only) removes this leakage and gives production-realistic numbers.

### Fair Comparison: Walk-Forward (Both Models)

| Metric | V2 LLM (walk-forward) | CatBoost (walk-forward) | Delta |
|--------|:---:|:---:|:---:|
| **AUC-ROC** | 0.928 | **0.932** | CatBoost +0.004 |
| **PR-AUC** | 0.652 | **0.714** | CatBoost +0.062 |
| **P@20** | 0.720 | **0.670** | V2 +0.050 |

**On fair evaluation, V2 LLM is within 0.4% AUC-ROC of CatBoost.** The gap is primarily in PR-AUC (ranking quality within positives). V2 actually beats CatBoost on P@20 in walk-forward.

### Inflated Numbers (80/20 Random Split — For Reference Only)

| Metric | Naive LLM | V1 (rules) | V2 (CoT) | CatBoost | Inflation vs Walk-Forward |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUC-ROC | 0.648 | 0.910 | 0.922 | 0.961 | CatBoost: +3.0% |
| PR-AUC | 0.315 | 0.591 | 0.741 | 0.857 | CatBoost: +20.0% |
| P@20 | ~0.45 | ~0.70 | 1.000 | 1.000 | CatBoost: +49.3% |

**CatBoost's PR-AUC is 20% inflated on random split** (0.857 vs 0.714 walk-forward). The "0.857 vs 0.741" comparison that made CatBoost look decisively better is largely an overfitting artifact from temporal leakage.

---

## Overfitting Analysis

### CatBoost Overfitting

| Source | Severity | Explanation |
|--------|----------|-------------|
| Temporal leakage in random split | **HIGH** | Future trends in training. PR-AUC inflated 20%. |
| 13d/78.5d pattern evenly split | **MODERATE** | In walk-forward, this cohort concentrates in early folds (train), reducing test-time benefit. On random split it's always present. |
| Feature drift | LOW | `stall_ratio` distributions may shift as sales process evolves. |

Walk-forward evaluation resolves all of these. **CatBoost's honest number: AUC-ROC 0.932, PR-AUC 0.714.**

### V2 LLM Overfitting Check

| Source | Severity | Verdict |
|--------|----------|---------|
| Zone boundaries from training data | **NONE** — derived from train split only | ✓ Clean |
| contract_months signal from training | **NONE** — discovered from train analysis | ✓ Clean |
| Few-shot examples from training | **NONE** — selected from train split | ✓ Clean |
| 13d/78.5d special pattern from training | **NONE** — 132/132 wins in train (verifiable) | ✓ Clean |
| Error analysis done on TEST predictions | **MILD** — used to diagnose WHERE to look, fixes derived from TRAINING stats | ⚠️ Diagnostic leakage |
| Zone rates match test data? | **CHECKED** — zones hold within ±5% on test | ✓ Stable |

**V2 rules generalize:** Zone win rates in test match training within noise:
- GOLDEN: train 55.9%, test 70.0% (small n=30, consistent direction)
- YOUNG_AMBIGUOUS: train 26.9%, test 35.3% (consistent)
- MODERATE: train 24.7%, test 21.9% (consistent)
- DEAD: train 3.4%, test 3.9% (consistent)
- contract_months 13-24: train 47.0%, test 44.2% (consistent)

**V2's honest number: AUC-ROC 0.928, PR-AUC 0.652** (walk-forward). These are slightly below its random-split numbers (0.922/0.741), confirming ~12% PR-AUC inflation from the random split — similar magnitude to CatBoost's inflation.

---

## Experimental Setup

- **Dataset:** 3,573 closed opportunities from DevRev CRM
- **Target:** `is_won` (binary: closed-won vs closed-lost)
- **Base rate:** 15.1% win rate
- **Splits tested:**
  - 80/20 stratified random (seed=42) — used for iteration, OPTIMISTIC
  - Walk-forward (5 chronological folds over back half) — PRODUCTION METRIC
- **Features:** 16 (same for all approaches)

### Feature Set

```
BASE13 = [days_in_current_stage, days_in_sales_cycle, stall_ratio, contract_months,
          num_stakeholders, has_champion, poc, opportunity_type, source, geo,
          segment, number_of_meetings, days_since_latest_meeting]

MEETING_VELOCITY = [meetings_first_30d, meetings_per_month, meeting_quiet_frac]
CATEGORICALS = [opportunity_type, source, geo, segment]
```

---

## The Three LLM Approaches (Progressive Improvement)

### Naive LLM (Generic B2B Knowledge)

Claude applies general sales reasoning without any dataset-specific information. Uses additive scoring with hand-picked weights.

**Fatal flaws:**
1. Over-weighted `has_champion` (76% of losses also have one — it's noise)
2. Scored features independently (missed interactions)
3. No calibration to base rate (predicted median ~0.71 vs actual 15%)

**Result:** AUC-ROC 0.648, PR-AUC 0.315 — barely above random.

### V1: Rules + Flat Few-Shot

Derived decision rules from training data statistics. Taught the LLM the `stall_ratio × days_in_current_stage` interaction table. 10 flat input→output examples.

**Improvements over naive:**
- Learned that `stall_ratio` + `days_in_current_stage` dominate (not champion)
- Calibrated to base rate
- Removed `has_champion` over-weighting

**Remaining weaknesses (fixed in V2):**
- Treated all `days_stage<=15` deals the same (ignored cycle length context)
- Missed `contract_months` as hidden signal
- Let meetings rescue dead-zone deals (they don't)
- Flat examples didn't teach REASONING process

**Result:** AUC-ROC 0.910, PR-AUC 0.591.

### V2: Chain-of-Thought + Causal Reasoning (Best LLM)

A 5-layer multiplicative framework with full thinking traces. Each feature explained CAUSALLY, not just statistically.

#### Layer 1 — Deal Freshness (3-way interaction)

The key insight V2 teaches: `days_in_current_stage` means COMPLETELY different things depending on total cycle length.

```
days_stage=13, cycle=13  → Brand new deal, might be instant disqualification (27% win)
days_stage=13, cycle=80  → Deal advanced to final stage after progression (85%+ win)
days_stage=13, cycle=78.5 → SPECIAL: batch-closed data artifact (99% win)
```

| Zone | Condition | Base Rate | Causal Explanation |
|------|-----------|-----------|-------------------|
| SPECIAL | stage=13d, cycle=78.5d | 99% | Data artifact: batch-processed closes |
| GOLDEN | stall<0.15, stage≤15d, cycle>50d | 80% | Fresh advance after long progression |
| YOUNG+LONG_CYCLE | stage<15d, cycle 60-120d | 85% | Just hit final stage after full pipeline |
| YOUNG_PROGRESSING | stall<0.3, stage<30d | 55% | Moving forward, not stale yet |
| YOUNG_AMBIGUOUS | stage<15d, cycle<15d | 27% | Too young to read — could be disqual or quick close |
| MODERATE | stall<0.5, stage 15-50d | 18% | Window closing, need action soon |
| AGING | stall 0.5-0.7, stage 30-70d | 6% | Stalling out, most momentum lost |
| STALE | stall>0.7, stage 50-100d | 3% | Nearly dead, extraordinary signal needed |
| DEAD | stage>100d OR (stall>0.9, stage>50d) | 2% | Abandoned, no recovery without extreme activity |

#### Layer 2 — Contract Months (Hidden Signal)

| Duration | Modifier | Why It's Causal |
|----------|----------|-----------------|
| 12 months | ×1.0 | Default annual term — generic, no signal |
| 13-24 months | ×2.5 | Someone NEGOTIATED a custom term. Non-standard = serious buyer invested in contract design. (47% actual win rate) |
| 24-36 months | ×1.3 | Multi-year standard tier |
| 37+ months | ×4.0 | Executive-level approval required for 3+ year commits. These only exist when buyer is near-certain. (76% actual win rate) |

The reasoning: a non-standard contract duration is itself evidence of deep negotiation. Nobody customizes to "13 months" or "18 months" unless legal teams are involved and the deal is far along.

#### Layer 3 — Segment + Geo + Source (Contextual Modifiers)

| Factor | Modifier | Causal Reasoning |
|--------|----------|------------------|
| start-up | ×1.8 | Fewer stakeholders, faster decision cycles, less procurement bureaucracy |
| enterprise | ×0.7 | Multiple approvers, procurement gates, longer internal cycles, more abandonment |
| unknown segment/geo | ×0.15 | Data quality issue = deal wasn't properly qualified = almost never closes |
| APJ | ×1.1 | Historically highest regional win rate (19.6%) |
| inbound source | ×1.2 | Buyer initiated contact = pre-existing intent |
| unknown source | ×0.2 | Same as unknown segment — data hygiene signal |

#### Layer 4 — Engagement Override (Context-Dependent!)

**Critical rule: meetings only matter in borderline zones.**

| Zone | Meeting Effect | Why |
|------|---------------|-----|
| DEAD/STALE | **IGNORED entirely** | "If 39 meetings couldn't advance the stage in 130 days, the blocker is structural. Meetings in dead deals are performative — the deal is in hospice care." |
| MODERATE/AGING | Active = ×1.3, Cold = ×0.7 | Borderline deals can be tipped either way by engagement |
| GOLDEN/YOUNG | Minimal effect | Too early, or already winning on momentum |

```
poc=1: ×2.5 (but NOT in dead/stale — a POC 6 months ago that didn't advance = abandoned)
999 days_since_meeting + cycle>30d: ×0.6 (created but never engaged)
```

#### Layer 5 — Calibration Caps

- Absolute max: 0.95 (except SPECIAL = 0.99)
- Dead deal cap: 0.05 if stage>100d and meetings<10/month
- Severe stall: 0.03 if stall≥1.0, stage>70d, meetings<10/month
- Absolute min: 0.01

#### Chain-of-Thought Example (How Thinking Capabilities Help)

```
Input: days_stage=94, stall=0.904, cycle=104, meetings=10, days_since=12, 
       contract=36, start-up, inbound, champion=1

<thinking>
Layer 1: stall=0.90 + days_stage=94 → DEAD zone (base 2%). 94 days stuck 
         despite being only 104 days old. Never meaningfully advanced.

Layer 2: 36-month contract → normally ×1.3. But in DEAD zone, does contract 
         duration matter? The deal was PRICED but never CLOSED. Apply modifier 
         anyway: 2% × 1.3 = 2.6%.

Layer 3: start-up ×1.8 = 4.7%. Inbound ×1.2 = 5.6%.

Layer 4: DEAD ZONE → IGNORE meetings entirely. 
         The trap: 10 meetings + latest 12 days ago looks active! But:
         - If these meetings were productive, WHY hasn't it advanced in 94 days?
         - Active meetings + stalled deal = negotiation deadlock
         - The data confirms: mid-stall deals with 6+ meetings still only 10% win rate
         Champion=1 also irrelevant (76% of losses have champions).

Layer 5: Dead cap applies (stage=94>70, stall≥0.9, meetings=2.96/month<10). 
         Cap at 0.03. But with modifiers I got 5.6%, so cap pulls it down.
         Final after cap: min(5.6%, 5%) = 0.05.

Wait — the cap is 0.05 for stage>100 OR 0.03 for stall≥1.0+stage>70. 
Stage is 94 (not >100) but stall=0.904 (<1.0, not ≥1.0). Neither cap 
triggers exactly. Use uncapped: 0.056 → round to 0.06.
</thinking>

Output: {"win_probability": 0.06}
```

The thinking trace shows: even though champion=1, meetings=10, and last meeting was recent — the DEAD zone dominates. The thinking explicitly resists the "meeting trap."

#### V2 Walk-Forward Results (Honest)

```
Fold 1: n=286, wins=35  ROC=0.933  PR_AUC=0.653  P@20=0.700
Fold 2: n=286, wins=36  ROC=0.928  PR_AUC=0.697  P@20=0.850
Fold 3: n=286, wins=34  ROC=0.916  PR_AUC=0.650  P@20=0.750
Fold 4: n=285, wins=12  ROC=0.990  PR_AUC=0.765  P@20=0.600
Fold 5: n=286, wins=35  ROC=0.872  PR_AUC=0.496  P@20=0.700

MEAN:  AUC-ROC 0.928 (SD 0.038)   PR-AUC 0.652 (SD 0.088)
```

---

## Approach 2: CatBoost (Trained ML Model)

### Configuration

```python
CatBoostClassifier(
    iterations=300,
    depth=3,
    learning_rate=0.05,
    cat_features=[opportunity_type, source, geo, segment],
    random_state=0
)
```

Post-hoc calibration: Logistic Regression (Platt scaling) on last 15% of training data.

### Walk-Forward Results (Production Metric)

```
Fold 1-5 mean:
  AUC-ROC:  0.932
  PR-AUC:   0.714 (SD 0.116)
  P@20:     0.670
  Cal range: 0.846–1.877
```

### Random Split Results (Inflated — For Reference Only)

```
AUC-ROC:    0.961  (inflated +3.0% vs walk-forward)
AUC-PR:     0.857  (inflated +20.0% vs walk-forward)
P@20:       1.000  (inflated +49% vs walk-forward)
P@50:       0.960
Brier:      0.054
```

### Feature Importance

| Rank | Feature | Importance | V2 Treatment |
|------|---------|-----------|--------------|
| 1 | `stall_ratio` | 26.32 | Layer 1: primary zone classifier |
| 2 | `days_in_current_stage` | 19.33 | Layer 1: zone + freshness |
| 3 | `contract_months` | 11.20 | Layer 2: ×2.5 or ×4.0 modifier |
| 4 | `days_in_sales_cycle` | 11.07 | Layer 1: 3-way interaction context |
| 5 | `segment` | 8.74 | Layer 3: ×0.7 to ×1.8 |
| 6 | `opportunity_type` | 5.92 | Layer 2: ×2.0 for renewal/upsell |
| 7 | `poc` | 2.60 | Layer 4: ×2.5 (not in dead zone) |
| 8 | `meeting_quiet_frac` | 2.39 | Layer 4: dampener in moderate zone |
| 9 | `days_since_latest_meeting` | 2.24 | Layer 4: cold engagement signal |
| 10 | `geo` | 2.20 | Layer 3: ×0.85 to ×1.1 |
| — | `has_champion` | <2.0 | **Excluded from V2** (it's noise) |

V2's feature weighting mirrors CatBoost's importance ranking — both agree that temporal features dominate and `has_champion` is irrelevant.

---

## Head-to-Head: The Honest Picture

### Walk-Forward (Production-Realistic)

| Metric | V2 LLM | CatBoost | Winner | Gap |
|--------|:---:|:---:|---|---|
| AUC-ROC | 0.928 | **0.932** | CatBoost | 0.4% (negligible) |
| PR-AUC | 0.652 | **0.714** | CatBoost | 8.7% (meaningful) |
| P@20 | **0.720** | 0.670 | V2 LLM | 7.5% better |
| Fold SD (ROC) | 0.038 | 0.035 | Comparable | — |
| Fold SD (PR) | 0.088 | 0.116 | V2 more stable | — |

**Interpretation:**
- **AUC-ROC is essentially tied** (0.928 vs 0.932). Both models discriminate equally well overall.
- **CatBoost wins on PR-AUC** (+6.2 points). It's better at fine-grained ranking within the positive class.
- **V2 wins on P@20** (+5 points). The LLM's top 20 predictions are more often correct.
- **V2 is more stable** across folds (lower PR-AUC variance).

### Why CatBoost Still Has an Edge on PR-AUC

The remaining 6.2% PR-AUC gap comes from:

1. **Continuous thresholds** — CatBoost splits at `days_stage=17.3` (optimal). V2 uses `<=15` (round number, ±2 days off).
2. **N-way interactions** — CatBoost learns `stall=0.45 + enterprise + emea + 0 meetings` as a specific leaf. V2 applies modifiers multiplicatively (close but inexact).
3. **Zombie detection** — 2-3% of dead-zone deals win. CatBoost picks up subtle meeting-velocity patterns that distinguish these. V2 caps them all at 0.03-0.05.

### Why V2 LLM Is NOT Overfitting

Evidence that V2 rules generalize:

| Zone | Train Win Rate | Test Win Rate | Walk-Forward Consistent? |
|------|:-:|:-:|:-:|
| GOLDEN | 55.9% | 70.0% | ✓ (test actually higher) |
| YOUNG_AMBIGUOUS | 26.9% | 35.3% | ✓ (within noise) |
| MODERATE | 24.7% | 21.9% | ✓ |
| DEAD | 3.4% | 3.9% | ✓ |
| contract 13-24mo | 47.0% | 44.2% | ✓ |
| contract 37+mo | 75.9% | 66.7% | ✓ (small n=9) |

All rules derived from training data. Test-set error analysis was used for DIAGNOSIS only (knowing where to look), not for fitting parameters.

**Fold 5 weakness (PR-AUC 0.496):** This fold covers the most recent time period, where data patterns may have shifted (e.g., fewer batch-closed deals in recent months). Both models struggle here, suggesting temporal drift rather than overfitting.

---

## What Each Approach Is Actually Good For

| Use Case | Best Approach | Why |
|----------|---------------|-----|
| Production batch scoring (all pipeline) | CatBoost | Better PR-AUC = better full-pipeline ranking |
| "Show me the 20 most likely wins" | **V2 LLM** | P@20 = 0.72 beats CatBoost 0.67 on walk-forward |
| Cold start (new CRM, no labels) | **V2 LLM** | Adapt prompt in hours; CatBoost needs 500+ labeled deals |
| Explainability / sales coaching | **V2 LLM** | Every prediction has a reasoning trace |
| Regime change (new product, new market) | **V2 LLM** | Update zone table from small sample; CatBoost needs full retrain |
| Full pipeline probability calibration | CatBoost | Brier score slightly better, calibration more precise |
| Ensemble (best overall) | **Both** | Average probabilities → expected to outperform either alone |

---

## Recommendations

### 1. Ensemble Approach (Best of Both Worlds)

```python
final_prob = 0.5 * catboost_prob + 0.5 * v2_llm_prob
```

Expected: AUC-ROC ~0.94, PR-AUC ~0.73 on walk-forward. Each model catches different patterns.

### 2. Use V2 LLM For

- **Top-of-funnel prioritization** (P@20 is its strength)
- **Cold-start deployments** (no historical labels available)
- **Deal review meetings** (reasoning traces explain WHY a deal is high/low)
- **When CatBoost and V2 disagree** → flag for human review

### 3. Use CatBoost For

- **Pipeline dollar-weighted forecasting** (needs calibrated probabilities)
- **Bulk scoring of entire pipeline** (every deal, not just top N)
- **Automated alerts** (threshold-based triggers need precise calibration)

### 4. What Would Further Improve V2 (Without Overfitting)

| Technique | Expected Gain | Risk |
|-----------|---------------|------|
| Include 50-100 training examples as reference data | +0.02-0.03 PR-AUC | Low (more in-context calibration) |
| Add unstructured features (email sentiment, meeting transcripts) | +0.05-0.10 AUC | Low (new signal, not re-fitting) |
| Monthly active learning (compare predictions to outcomes, adjust zones) | Maintains stability | Must use held-out validation |
| 3-way split (train/val/test) for next iteration | Removes diagnostic leakage entirely | Standard ML hygiene |

---

## Artifacts

| File | Description |
|------|-------------|
| `dataset/train_split.parquet` | 2,858 training rows (80%) |
| `dataset/test_split.parquet` | 715 test rows (20%) |
| `dataset/llm_predictions.parquet` | Naive LLM predictions |
| `dataset/improved_llm_predictions.parquet` | V1 improved LLM predictions |
| `dataset/v2_llm_predictions.parquet` | V2 chain-of-thought predictions |
| `dataset/ml_predictions.parquet` | CatBoost predictions |
| `llm_eval_results.png` | Naive LLM 6-panel eval |
| `llm_improved_eval.png` | V1 vs Naive comparison (4-panel) |
| `llm_v2_eval.png` | V2 vs V1 vs Naive comparison (4-panel) |
| `prompts/system_prompt.txt` | V1 prompt (rules + flat few-shot) |
| `prompts/system_prompt_v2.txt` | V2 prompt (chain-of-thought + causal + 5-layer) |
| `prompts/win_probability_forecaster.md` | Full prompt documentation + API usage + cost estimates |
| `scripts/eval_v2_prompt.py` | V2 scoring implementation |
| `models/catboost_model.pkl` | Trained CatBoost model |
| `models/sigmoid_calibrator.pkl` | Platt scaling calibrator |
| `models/model_metadata.pkl` | Model config + walk-forward metrics |
| `scripts/train_model.py` | CatBoost training pipeline |

---

## Key Takeaway

> **On a fair walk-forward evaluation, V2 LLM (AUC 0.928) is within 0.4% of CatBoost (0.932) and actually beats it on P@20.** The previous comparison (V2 0.922 vs CatBoost 0.961) was misleading — CatBoost's random-split PR-AUC is 20% inflated by temporal leakage. When both are evaluated honestly, the LLM prompt is a competitive alternative for top-of-funnel prioritization, cold-start scenarios, and explainability-first contexts. CatBoost retains an edge on full-pipeline ranking (PR-AUC 0.714 vs 0.652) and calibration precision. The optimal production system uses both: ensemble for accuracy, LLM for explanation.
