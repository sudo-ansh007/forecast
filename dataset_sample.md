# forecast_v0 — Dataset, Fields, Metrics

Generated 2026-08-10 from `dim_opportunity` (12,963 deals). Deal IDs are real.

---

## 1. The 11 fields we train on

| Our name | Source field | Type | Means |
|---|---|---|---|
| `days_in_current_stage` | `tnt__days_in_current_stage` | number | Days since the stage last changed |
| `days_in_sales_cycle` | `tnt__days_in_sales_cycle` | number | Days since the deal was created |
| `stall_ratio` | *derived:* `days_in_current_stage / days_in_sales_cycle` | 0–1 | Share of the deal's life parked in one stage |
| `contract_months` | `tnt__contract_months` | number | Contract term |
| `num_stakeholders` | `len(contact_ids)` | count | How many contacts are on the deal |
| `has_champion` | `tnt__champion` filled? | 0/1 | A champion is named |
| `poc` | `tnt__poc` | 0/1 | POC **recorded** — checkbox, not evidence a POC ran (see §6) |
| `opportunity_type` | `tnt__opportunity_type` | category | new / renewal / upsell / amendment |
| `source` | `tnt__source` | category | outbound / inbound / events / partner / advisor / referral / unknown |
| `geo` | `tnt__geo` | category | amer / apj / emea / unknown |
| `segment` | `tnt__segment` | category | enterprise / mid-market / smb / start-up / 2k / unknown |

Two derived names worth knowing: **`stall_ratio`** is ours, not a CRM field —
it's the strongest signal we have. **`has_champion`** is a yes/no we compute from
whether the champion field is filled; we never use the person's ID.

Carried alongside but **not** trained on: `display_id` (join key), `acv` (revenue
roll-up only), `stage_name` (gives the label on closed deals), `is_won` (the label).
The sample tables below still show an `nda` column — shown for reference, **not a
feature**; it was dropped (§5).

---

## 2. Sample rows

### Won — `is_won = 1`

| display_id | stall_ratio | days_in_stage | days_in_cycle | contract_mo | stakeholders | champion | poc | nda | type | source | geo | segment | acv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPP-4450 | **0.014** | 1 | 70 | 12 | 1 | **1** | 0 | 0 | new | outbound | apj | start-up | $6,920 |
| OPP-1286 | **0.031** | 2 | 64 | 13 | 2 | **1** | 1 | 1 | new | outbound | emea | start-up | $6,366 |
| OPP-100 | 0.166 | 13 | 78.5 | 15 | 0 | 0 | 0 | 0 | new | outbound | amer | smb | $768 |
| OPP-319 | 0.166 | 13 | 78.5 | 12 | 1 | **1** | 0 | 0 | new | events | amer | start-up | $240 |
| OPP-58 | 0.166 | 13 | 78.5 | 15 | 2 | 0 | 0 | 0 | new | events | amer | smb | $21,064 |
| OPP-8061 | 0.166 | 13 | 78.5 | 12 | 1 | **1** | 0 | 0 | upsell | outbound | amer | smb | $4,217 |
| OPP-2056 | 0.261 | 43 | 165 | 36 | **7** | **1** | 1 | 1 | new | outbound | amer | mid-market | $108,333 |
| OPP-2546 | 0.500 | 1 | 2 | 12 | 1 | **1** | 0 | 0 | new | partner | apj | start-up | $5,038 |

### Lost — `is_won = 0`

| display_id | stall_ratio | days_in_stage | days_in_cycle | contract_mo | stakeholders | champion | poc | nda | type | source | geo | segment | acv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPP-6288 | 0.415 | 97 | 234 | 12 | 1 | 0 | 0 | 0 | new | inbound | apj | smb | $20,280 |
| OPP-7588 | 0.556 | 15 | 27 | 12 | 1 | 1 | 0 | 0 | new | partner | amer | smb | $25,000 |
| OPP-11390 | 0.600 | 21 | 35 | 36 | 1 | 1 | 1 | 0 | new | outbound | amer | mid-market | $50,000 |
| OPP-5058 | 0.698 | 104 | 149 | 12 | 1 | 0 | 0 | 0 | new | outbound | apj | enterprise | $50,000 |
| OPP-1000 | 0.828 | 53 | 64 | 12 | 3 | 1 | 0 | 0 | new | outbound | amer | smb | $25,000 |
| OPP-253 | 0.828 | 53 | 64 | 12 | 0 | 0 | 0 | 0 | new | events | amer | mid-market | $25,000 |
| OPP-13537 | **1.000** | 7 | 7 | 12 | 1 | 0 | 0 | 0 | new | outbound | apj | 2k | $100,000 |
| OPP-8739 | **1.000** | 63 | 63 | 12 | 1 | 0 | 0 | 0 | new | events | apj | mid-market | $40,000 |
| OPP-3850 | **1.000** | 100 | 100 | 12 | 1 | 0 | 0 | 0 | new | partner | amer | enterprise | $50,000 |

### Open — what we score, no label

| display_id | stall_ratio | days_in_stage | days_in_cycle | contract_mo | stakeholders | champion | poc | nda | type | source | geo | segment | acv | stage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OPP-10006 | 0.459 | 73 | 159 | 36 | 2 | 1 | 1 | 0 | new | outbound | amer | smb | $117,667 | 5-validate |
| OPP-13098 | 0.868 | 33 | 38 | 12 | 1 | 1 | 0 | 0 | new | outbound | amer | 2k | $587,500 | 3-evaluate |
| OPP-11236 | 0.872 | 95 | 109 | 12 | 3 | 1 | 0 | 1 | new | outbound | amer | enterprise | $100,000 | 2-qualify |
| OPP-13802 | 1.000 | 10 | 10 | 12 | 1 | 0 | 0 | 0 | new | outbound | apj | enterprise | $55,000 | 1-profile |
| OPP-13467 | 1.000 | 25 | 25 | 12 | 1 | 0 | 0 | 0 | new | inbound | amer | start-up | $6,920 | 1-profile |
| OPP-12688 | 1.000 | 57 | 57 | 12 | 1 | 0 | 0 | 0 | new | outbound | amer | smb | $20,280 | 1-profile |
| OPP-12052 | 1.000 | 77 | 77 | 12 | 1 | 0 | 0 | 0 | new | inbound | amer | start-up | $6,920 | 1-profile |
| **OPP-4270** | 1.000 | **465** | 465 | 12 | 0 | 0 | 0 | 0 | renewal | partner | apj | enterprise | $170,463 | 1-profile |

Sorted by `stall_ratio` so the pattern reads top-to-bottom: won deals sit low,
lost deals sit high. **6 of 8 won deals have a champion; 4 of 9 lost do.**

**OPP-4270** is the case today's forecast misses: 465 days in stage 1, no
champion, no contacts, $170k enterprise renewal — booked at full stage-constant
value right now.

---

## 3. Population & split

| | Count |
|---|---|
| Total deals | 12,963 |
| Closed — trainable | 11,051 |
| — **won** | **541** (4.9%) |
| — lost | 10,510 |
| Open — scored | 1,912 |

Split by `created_date`: oldest 80% train (8,840 / 492 wins), newest 20% test
(2,211 / 49 wins). Never random — a random split lets future outcomes leak backwards.
This is not cosmetic: splitting `ml_train.csv` in its raw row order (which is *not*
chronological) lands 78 test wins and PR-AUC 0.75 instead of 0.57. Sort by
`created_date` before splitting.

---

## 4. Metrics — what we measure and why

Win rate is 4.9%, so **accuracy is useless**: predicting "everything loses"
scores 95.1% and finds nothing.

| Metric | Now | Target | What it tells you |
|---|---|---|---|
| **PR-AUC** | **0.5744** | > 0.25 | Ranking quality. Random = 0.0222, so we're 26× baseline. Primary metric |
| **Brier** | **0.0139** | lower than rep-entered `probability` | Calibration. **The one that matters** — the product is a revenue number, and 30% has to mean 30% |
| **Recall @ top 20%** | **0.939** | > 50% | Of deals that actually won, the share in our top-scoring fifth. How a manager reads it: review 20% of pipeline, catch 94% of real wins |
| **ROC-AUC** | 0.9621 | reported only | Do **not** optimise — inflated by 2,162 easy negatives |
| **Σ probabilities vs actual wins** | 1.33× | 0.9–1.1× | Aggregate sanity. We over-predict by 33% — calibrator needs a retune before the roll-up is quotable |
| **Reliability curve** | see below | ±10% per bin | Predicted vs actual per probability band |

### Reliability, 10 bins

| bin | deals | predicted | actual |
|---|---|---|---|
| 0–10% | 2,067 | 1.0% | 0.6% |
| 10–20% | 63 | 13.7% | 7.9% |
| 20–30% | 27 | 24.2% | 11.1% |
| 30–40% | 20 | 33.6% | 35.0% |
| 40–50% | 6 | 45.0% | 33.3% |
| 50–60% | 6 | 57.6% | 50.0% |
| 60–70% | 8 | 65.9% | 87.5% |
| 70–80% | 2 | 72.5% | 0% |
| 80–90% | 9 | 85.0% | 77.8% |

The 0–10% bin holds 2,067 deals and is trustworthy. Everything above it holds
2–27 deals — the wobble there (including the 70–80% bin reading 0%, which is two
deals both losing) is small-sample noise at 49 test wins, not miscalibration.

### Baselines we still owe

| Baseline | Status |
|---|---|
| Rep-entered `probability` | Not comparable — it's 100 on won deals, 0 on everything else. A record, not a forecast |
| **Stage-constant × amount** (today's forecast) | **Can't measure yet.** On closed deals the stage *is* the label. Needs point-in-time history. This is the ship/no-ship gate |

---

## 5. Fields we deliberately left out

| Field | Why |
|---|---|
| `amount` / `acv` | Filled *after* signing — 93.2% of won vs 28.8% of lost. Bigger deals win *less* (AUC 0.334). Used only in the revenue roll-up |
| `technical_voc` | 8.1% of closed deals vs **0.1% of open** — field stopped being maintained. Would train on 896 examples then never appear again |
| `created_date` (raw) | Old cohorts have resolved, recent ones haven't. Win-rate-of-closed runs 94.6% (2023Q2) → 2.5% (2026Q2) purely from that. Adding it drops PR-AUC to 0.34 |
| `target_close_date` | Overwritten to the actual close date on 99.3% of closed deals. Also missing on 44.7% of open ones |
| `stage` | On a closed deal, stage *is* the label (8 = won, 9 = lost) |
| `probability`, `forecast_category`, `tnt__ramped_arr`, `tnt__win_notes`, `tnt__closed_lost_reasons`, `tnt__agentic_score`, `ctype__*` | Filled *because* the deal closed. Pure leakage |
| `nda` | Strong raw lift (40.6% win rate on the 138 closed deals with it vs 4.4% without) but only **56 wins** behind it — a shallow tree carves a near-pure leaf and overfits. Dropping it *improves* PR-AUC 0.5631 → 0.5781 across 5 seeds, a gap 2.5× the seed spread |
| `has_se`, `meddpicc_count` | Real signal but fill diverges by outcome; `has_se` adds only +0.003 on top of `has_champion` |

`source` is the weakest field still in the set: 1.0%–7.4% spread, and only 2× between
the two levels carrying volume (outbound 5.19% / inbound 3.39%). Tiny cells — customer
referral 3 deals, advisor 3 — and `unknown` at 1.02% is a hygiene artifact, not a
channel. Worth ablating; cut it if PR-AUC moves less than the seed spread.

---

## 6. Known limits

**Numbers above are optimistic.** We train on closed deals in their *final*
state but score open deals *mid-flight*. `days_in_sales_cycle` means "final cycle
length" on a closed deal and "age so far" on an open one — same column, two
meanings. Sizing that gap needs backtesting we can't do yet.

**`has_champion` direction is real, magnitude isn't.** Won deals have one 81.7%
of the time vs 39.3% for lost — partly because champions help, partly because
reps backfill the field on deals they're already closing.

**541 wins is the binding constraint**, not model choice.

**`segment=2k` is a real segment the model can't score yet.** 1 win in 111 closed deals
but **10% of open pipeline**. Checked whether it's a data-entry artifact — it isn't: 302
deals across 139 distinct creation dates from 2025-02-17 onward, only 19 bulk-imported,
$150k median ACV. It's simply *young*, so 185 of 302 are still open and the only 2k
deals that have resolved are the fast losses. Its 0.90% win rate is cohort censoring
inside one segment, not a real rate. Don't quote 2k probabilities until the cohort
matures.

**Two more levels have no training support.** `segment=unknown` has **0 wins in 1,174**
closed deals and is 4% of open. `geo="public sector"` appears on 3 open deals and never
on a closed one — one-hot encoding zeroes it silently, so those deals score as if geo
were blank. `make_dataset.py` warns on each; treat predictions there as unsupported.

**Calibration always extrapolates forward.** 89% of scored deals (1,709 of 1,912) were
created in 2026, while most training labels are 2024–2025 — a resolved cohort can't be
recent, by definition. No choice of window fixes it; it's why Brier and the reliability
curve matter more than PR-AUC here.

**Renewals and new business win very differently** — renewal 28.4%, upsell 20.0%, new
3.8%, an 8.8× spread wider than any other field, with 80.6% of open deals `new`. The
obvious fix is to split the dataset. **Measured, and it's worse:** training on new only
costs PR-AUC 0.5781 → 0.5536, 4× the seed spread. `opportunity_type` is already a
feature, so the tree carves the renewal branch itself where it pays, and pooling keeps
those 126 renewal/upsell wins contributing to every *other* split. Splitting discards a
quarter of the label supply to isolate a distinction the model had already priced.
Revisit at ~300 renewal wins, enough to fit standalone. Same story for restricting to
the 2024Q1–2025Q4 cohort window: nominally +0.013 PR-AUC but with 3× the spread, and
calibration degrades 1.36× → 1.54×. Both flags exist in `build_features.py`, both off,
numbers recorded there.

**Imputed amounts are a shrunk geometric mean, not a median.** Averaged in log space
by segment × opportunity_type, then shrunk toward the global value by `n/(n+20)` so
thin groups don't invent a number off 3 deals. Imputed values span $3.0k (start-up
new) to $52.6k (enterprise new) — 17.4×, monotone by segment. Separately, **130 deals
carry a real ACV under $10** (down to $0.03), which looks like a units error at entry;
harmless here since `acv` isn't a feature, but worth flagging to CRM owners.

**`poc` means "POC recorded", not "a POC ran."** Checkbox with no unset state
(362 True / 12,601 False / 0 null), and 165 deals have POC *notes* written with the
box still False. The True side carries real signal (41% win rate on the 56 flagged
test deals); the False side conflates "no POC" with "nobody filled it in".

**Four features blocked on missing history** (`fact_opportunity`'s log starts
2026-05-01; deals go back to 2023-01): stage-at-time, close-date push count,
champion-identified date, and honest backtesting. All unlocked by one thing —
starting a daily `dim_opportunity` snapshot.
