# `win_probability_colab.ipynb` — presenter's walkthrough

Companion to the notebook. For each section: what it does, what to say, which number
is the one that matters, and what "good" versus "bad" looks like on that number.

Read §0 and §1 before presenting. If you only have ten minutes with an audience, present
§4 (Section 4), §9 (Section 5) and §10 (Section 6) and skip the rest.

Version: `v3-2026-08-11-xgboost-evidence-gate`. Commit `3c4a6da`.

---

## 0. What this replaces, in one paragraph

DevRev's current forecast is `deal amount × a stage probability an admin typed in`.
Every deal at stage 4 gets the same number, so the forecast cannot tell a healthy stage-4
deal from a stalled one, and it only moves when a rep drags a card. This model reads 11
fields already on the deal and outputs a per-deal probability that is *calibrated*: when
it says 30%, roughly 30% of those deals close.

**One thing to be straight about up front.** We cannot measure whether this beats the
incumbent, and it is not for lack of trying. On a closed deal the stage *is* the outcome
— every won deal reads `8-closed_won`, every lost one `9-closed_lost`:

```
stage_name         lost   won
8-closed_won          0   541
9-closed_lost      3032     0
```

So scoring the stage-constant forecast on historical deals gives it a perfect 1.0. It's
not a real 1.0, it's a tautology. A true head-to-head needs point-in-time stage history,
which doesn't exist yet (§12). Anyone who asks "is it better than what we have?" deserves
that answer, not a number.

What we *can* say: against no model at all, on 715 unseen deals with 79 wins,

| | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| this model | **0.630** | **0.879** | **0.067** |
| flat base rate (no model) | 0.110 | 0.500 | 0.101 |

---

## 1. Two things to say before any number

Say both of these out loud early. If a sharp person finds them on their own after you
quoted a dollar figure, the whole thing looks shakier than it is.

**a) The dollar total is soft; the ranking is solid.** ACV is real on 41% of open deals
and imputed on the rest. Lead with the ranking.

**b) The two strongest features are partly measuring the wrong thing.**
`days_in_current_stage` (importance 0.206) is frozen at close on a training row but live
on an open deal. `stall_ratio` pins at 1.0 for 72.8% of open deals versus 19.3% of closed
ones — same column, two distributions. That means **0.879 ROC is optimistic; it is not
the production number.** Ranking within the open pipeline still works. Precise
probabilities are borrowed from a distribution the pipeline doesn't share.

This is fixable, it needs daily snapshots, and it's the top ask (§12).

---

## 2. How to run it

**Colab (what to use for a demo).** Open from the badge in cell 0, `Runtime → Run all`,
upload `ml_train.csv` and `ml_score.csv` when prompted. ~4 minutes.

**Locally.**

```bash
export DEVREV_TOKEN=...          # never commit or echo this
python build_features.py         # dim_cache.parquet -> train.parquet + score.parquet
python make_dataset.py           # -> ml_train.csv, ml_score.csv, ml_predictions.csv
```

`make_dataset.py` is the shipping path and prints the same forecast the notebook does.
The notebook is the *explanation* of that path, not the path itself.

**Confirming you're on the current version.** Three guards, because a stale run is the
failure mode that looks like success — PR-AUC lands near 0.62 on the old dataset too,
since PR-AUC's floor *is* the base rate.

1. Cell 3 prints `notebook v3-2026-08-11-xgboost-evidence-gate`.
2. Cell 4 prints `data OK for v3-...`, or raises `SystemExit` unless the CSVs are exactly
   3,573 train rows / 541 wins / 1,912 score rows with an `evidence` column.
3. Cell 4 checks the CSV *shape*, not the filename, before skipping the upload. Colab's
   `/content` survives a runtime restart, so a filename check finds the previous run's
   files and silently scores the old data. Mismatched files get deleted and re-requested.
   `FORCE_UPLOAD = True` forces it.

**If cell 4 raises `SystemExit`:** re-export from the repo and re-upload. Don't work
around it — that exception is the guard doing its job.

---

## 3. Sections 1–3 — setup, features, encoding (cells 2–10)

Skip in a short presentation. Two things worth knowing if asked.

**What goes in: 11 features → a `(3573, 28)` float matrix.**

| # | feature | type | source | importance |
|---|---|---|---|---|
| 1 | `days_in_current_stage` | numeric | `tnt__days_in_current_stage` | **0.206** |
| 2 | `stall_ratio` | numeric | derived: #1 ÷ #4 | **0.114** |
| 3 | `contract_months` | numeric | `tnt__contract_months` | **0.096** |
| 4 | `days_in_sales_cycle` | numeric | `tnt__days_in_sales_cycle` | 0.015 |
| 5 | `num_stakeholders` | numeric | `len(contact_ids)` | 0.011 |
| 6 | `has_champion` | 0/1 | `tnt__champion` is filled | 0.002 |
| 7 | `poc` | 0/1 | `tnt__poc == true` | 0.030 |
| 8 | `opportunity_type` | categorical → 4 cols | `tnt__opportunity_type` | 0.021 |
| 9 | `source` | categorical → 7 cols | `tnt__source` | — |
| 10 | `geo` | categorical → 4 cols | `tnt__geo` | 0.010 |
| 11 | `segment` | categorical → 6 cols | `tnt__segment` | 0.042 |

Nine come straight from DevRev custom fields. Only `stall_ratio` and `num_stakeholders`
are ours. That's deliberate: no new data collection to deploy this.

**What is deliberately NOT a feature**, and why each would be a leak:

- `acv` — filled at signing. Revenue weight only.
- `created_date` — carries cohort censoring. Split key only.
- `stage_name` — the label on closed deals. Exported to `ml_score.csv` only, never to
  `ml_train.csv`, so nobody can add it to `FEATURES` by accident.
- `probability` — the incumbent's own output. 11,558 of 12,963 deals have it at 0.0.
- `evidence` — a reporting gate, not a signal (§9).

---

## 4. Section 4 — honest evaluation (cells 11–13)

**What it does.** Sorts by `created_date`, trains on the oldest 80% (2,858 deals / 462
wins), tests on the newest 20% (715 / 79). A time split, because a random split lets a
deal's own future neighbours into training and inflates PR-AUC from 0.56 to 0.75.

**The five numbers, and how to read each.**

| metric | value | what it means | good | bad |
|---|---|---|---|---|
| **PR-AUC** | 0.630 | ranking quality on the rare class | ≥5× base rate | near base rate (0.110) |
| **Brier** | 0.067 | mean squared probability error | < base-rate model's 0.101 | above it |
| **ROC-AUC** | 0.879 | separation, base-rate invariant | > 0.80 | < 0.70 |
| **Recall@20%** | 0.785 | share of wins in the top-scoring fifth | > 0.60 | ≈ 0.20 |
| **Σp / actual** | 1.10× | roll-up sanity | 0.9–1.1× | > 1.3× or < 0.7× |

**Recall@20% = 0.785 is the number to present.** Everything else is a modelling metric;
this one is a workflow statement. *If a rep works the top 20% of scored deals, they touch
79% of the deals that will close.* Random gets 20%. That's 3.9× lift, and it's what the
model is actually for.

```
  top   5%  0.342   random 0.050   lift 6.8x
  top  10%  0.532   random 0.100   lift 5.3x
  top  20%  0.785   random 0.200   lift 3.9x
  top  50%  0.911   random 0.500   lift 1.8x
```

**Two traps.**

*PR-AUC is not comparable across datasets.* Its floor **is** the base rate. This dataset
moved from 2.2% to 11.0% base rate, so every PR-AUC number in any older document or
commit is void as a comparison. Use lift (PR-AUC ÷ base rate) or ROC-AUC across sets.

*Accuracy is omitted on purpose.* At an 11% win rate, "predict everything loses" scores
89% and finds zero deals. The cell prints this so nobody asks.

**Cell 13, the reliability table** — the one that decides whether the dollar figure is
usable at all. Predicted vs actual, decile bins:

```
        bin  deals  predicted  actual
0    0%-10%    582     0.0351  0.0344      <- 81% of deals, near-exact
1   10%-20%     30     0.1444  0.2667
9  90%-100%     21     0.9361  0.9524      <- the confident end holds
```

Read it: **the 0–10% bin holds 582 of 715 deals and is accurate to 0.0007.** The
90–100% bin holds and is also close. Both ends work.

**What to concede if asked.** The middle bins (0.25–0.85) sit *below* the diagonal —
0.56 predicted came in at 0.38 actual, 0.85 at 0.50. Each holds only 6–18 deals so
individually it's noise, but the direction is consistent across seven bins, and noise
usually isn't directional. Honest statement: **the roll-up is right and mid-range
individual scores lean optimistic.** A deal shown at 60% is worth less than 60%.

---

## 5. Section 4b — is it overfitting? (cells 14–15)

**What it does.** Splits the train/test gap into its two causes, which a raw gap conflates.

```
               split     n  wins  base   PR-AUC  lift   ROC
     train in-sample  2858   462  16.2%  0.9166   5.7  0.9756
  train OUT-of-fold   2858   462  16.2%  0.8598   5.3  0.9560
   test (time split)   715    79  11.0%  0.6300   5.7  0.8792
```

In-sample → OOF holds the base rate fixed, so that gap (**+0.057**) is overfitting.
OOF → test changes the era, so that gap (**+0.230**, 4× larger) is distribution shift.

**The shift is cohort censoring, not memorisation.** Test deals are the newest, so the
youngest. Among young deals only the fast losses have resolved; their eventual wins are
still open and sitting in `ml_score.csv`. That is the entire 16.2% → 11.0% move.

Normalised for base rate the model discriminates *slightly better* on test: 5.3× lift
out-of-fold vs 5.7× on test.

**Correction worth knowing, because an older version of this deck said the opposite.**
On the pre-filter dataset the *least* regularised config also scored best on test, which
said the gap measured nothing. That no longer holds. Measured now: `depth8/mcs5` fits
train perfectly (in-sample 1.000) and is **last** on test at 0.545. So *some* of the gap
is real overfitting. The 0.057 in-sample→OOF gap still bounds it and is 4× smaller than
the shift, so censoring dominates — but the sweep no longer proves overfitting is absent,
only that regularising past the current setting doesn't help.

---

## 6. Sections 4c–4e — why this model (cells 16–21)

**What it does.** 26 configurations, 5 seeds each, same time split, same calibrator.
Includes a logistic-regression floor and a single decision tree, so the boosters have
something to beat.

```
                     model  PR-AUC   sd    Brier    ROC   Σp/act
      XGBoost  SHIPPING    0.6300  0.0000  0.0668  0.8792  1.10
    CatBoost d3 it600      0.6191  0.0027  0.0676  0.8766  1.14
    LightGBM (was ship)    0.6163  0.0000  0.0685  0.8773  1.10
    random forest 500      0.5315  0.0020  0.0738  0.8482  1.15
    logreg L2 (not a tree) 0.4944  0.0000  0.0793  0.8495  1.32
    single tree depth3     0.4457  0.0000  0.0757  0.8114  1.12
```

**Why XGBoost.** It wins all seven measured columns — ranking, discrimination, per-deal
error, calibration error, top-quintile recall and aggregate ratio. No trade-off to argue
about. Both it and LightGBM have seed sd 0.0000 (neither subsamples by default), so the
+0.014 gap is not seed luck. The forecast barely moves: $4.04M vs LightGBM's $3.97M.

**Say this if asked "did you tune it?"** Yes, and tuning *lost*. A 48-config grid scored
by out-of-fold PR-AUC within train picked a config that then dropped 0.015 on test. With
79 test positives, chasing 0.01 differences is fitting noise. Depth 3 / 200 trees / lr
0.05 is deliberately boring.

**Calibration is a units fix, not free accuracy** — the cleanest counter-example in the
whole notebook. CatBoost *uncalibrated* ranks mid-pack (0.6176) yet ships **$0.6M more
pipeline** than the calibrated models, with a max probability of 0.880 vs 0.954. Its
ordering looks fine and its numbers are unusable as revenue weights. That's why every
candidate goes through `CalibratedClassifierCV`.

**One unresolved discrepancy, in the interest of not hiding it.** CatBoost-raw measured
0.630 in an earlier notebook run and 0.6159 in `pick_shipping.py` on what should be the
same config and data; this run gave 0.6176. A ~0.014 swing, larger than the gap that
decided the shipping model. It doesn't change the XGBoost pick (XGBoost wins on Brier,
ECE and calibration regardless) but it is not explained.

---

## 7. Section 2b — look at the data first (cells 7–8)

Two things that look like bugs and are not. Have these ready; they get spotted.

- **`stall_ratio` piles at exactly 1.0.** The deal never left its first stage — not a cap
  artifact. Strongest single split in the dataset.
- **`num_stakeholders == 0` on 7% of train, 13% of open.** Not imputed, genuinely blank.

---

## 8. Section 4d — the same models as pictures (cells 18–19)

Six panels. Two matter for a presentation:

**Reliability** — catches a model that PR-AUC calls excellent while it over-promises.
This is the panel to show if someone challenges the dollar figure.

**PR-AUC vs calibration error** — bottom-right is what you want: ranks well *and* the
roll-up is right. It shows the two objectives are not the same axis.

---

## 9. Section 5 — the forecast (cells 22–24)

Refits on all 3,573 closed deals and scores the 1,912 open ones. **The headline output.**

```
1,912 open deals

COMMITTED (evidence >= 2)  622 deals  58 expected wins  $4.0M
  win_probability   min 0.029  median 0.036  max 0.952
TOO EARLY                1,290 deals  win_probability NULL
  (would have added 65 wins / $1.9M -- excluded on purpose)

health bands
Green          41
Yellow         72
Red           509
Too early    1290
```

**The evidence gate is the part to explain carefully — it's a judgment call, not a
metric.** `evidence` counts how many independent things a rep actually recorded:
champion, POC, ≥2 contacts, advanced at least one stage, real ACV. Below 2, the deal has
nothing on it, and the matching closed cohort wins 2.1% of the time versus 16.9% at
evidence >= 2 (22.8% at exactly 2). The model is reading *absence of data*, so its output there isn't a
forecast.

Three specific choices inside that, each with a reason:

**It gates on evidence, not stage.** A 1-profile deal with champion, POC and stakeholders
filled in clears the gate and keeps its score — early-but-documented is exactly the
signal worth having. Only absence is censored. (You can see this in the output: a
`1-profile` deal sits at 0.930, second-highest in the file.)

**Probability is NULLed, not lowered.** A low number in a CSV cell gets summed by whoever
opens it. Totalling all 1,912 gave **$5.67M against a committed $4.1M** — 1,290 unworked
deals at ~0.03 each add up. The band alone didn't stop that; an empty cell does.
`raw_probability` keeps the model's actual output for auditing, under a name nobody
totals by accident.

**Rows are gated, not deleted.** Deleting 60% of pipeline from a forecast is worse than
labelling it. The deals are still in the file, visible, marked `Too early`.

**How to check the gate is working**, in the cell 27 chart: the `Too early` bar reads
**$0.00 across 1,290 deals**. Nothing unworked is in the roll-up. Four asserts enforce it
and will fail the run:

```python
assert out.raw_probability.between(0, 1).all()
assert out.health.notna().all()
assert out.win_probability.isna().equals(too_early)          # NULLs match the gate
assert out.loc[too_early, "expected_revenue"].isna().all()   # no revenue on a gated deal
```

**Expect the $4.0M / $4.1M difference.** The notebook refits on all of train; the
pipeline's calibration folds land differently. Same numbers to one decimal, and if they
ever diverge by more than that, the two paths have drifted.

---

## 10. Section 6 — what a sales manager acts on (cells 25–27)

Where the model stops being a metric. Present this.

**The at-risk list**: > $50k, < 10% win probability, sorted by longest stalled.
**$3.2M across 15 deals**, every one currently booked at full stage-constant value by
today's forecast. That is the concrete argument for the whole project.

**Say the caveat that's printed under the table.** If those deals share a created-date
cluster and all have 0 stakeholders, they're a bulk import that was never worked — a
data-hygiene story, not a rep-behaviour story. Different fix, different audience. Check
before presenting.

**Health bands are placeholders.** Red/Yellow/Green at 0.10/0.30 was chosen by taste, not
measurement. Sales owns those thresholds and should set them from the reliability table.
Say so before someone assumes they're validated.

---

## 11. Section 7 — limits (cell 28)

Already written in the notebook, in full. Read it before presenting; don't summarise it
away. The five that come up most:

1. **Metrics are optimistic** — the train/serve skew in §1b.
2. **Revenue is softer than the probability** — ACV real on 41%, rest imputed as a shrunk
   geometric mean by segment × type. Range, never a point.
3. **`segment=2k` is real but unscoreable** — 1 win in 22 closed deals post-filter, 10% of open
   pipeline. Its 0.90% win rate is censoring, not a rate. `encode()` warns about it at
   runtime.
4. **541 wins is the binding constraint**, not model choice. Deeper models won't help.
5. **Single-tenant.** Learns one company's patterns. Ranking probably transfers; base
   rates definitely don't.

Also in there: `has_champion`'s magnitude is inflated by reps backfilling the field on
deals they're already closing; `poc` is a checkbox with no unset state, so its False side
conflates "no POC" with "nobody filled it in"; `segment_unknown` scoring as a driver
means the model is partly learning CRM hygiene rather than deal health.

---

## 12. Known issues and the one ask

**Two features are partly artifacts.** Both are in the top 3.

*`contract_months` (0.096) — post-close artifact, fixable in one line.* Off-round terms
(13, 14, 15, 27 months) win **69.9%** vs 11.4% on round terms, and carry 29.6% of all
training wins. They are 0.4% of open pipeline — 7 deals of 1,912. An off-round term gets
written at signing when someone negotiates a mid-month start; it's a consequence of
closing, not a predictor.

Scored on round-term test deals only — which is what the pipeline actually looks like —
the feature is worth **+0.025, not +0.065**. Roughly 60% of its apparent value comes from
15 test deals with no counterpart in the pipeline being forecast.

| | PR-AUC | ROC | ships |
|---|---|---|---|
| current | 0.6304 | 0.8790 | $4.04M |
| collapse to nearest 12mo | 0.5784 | 0.8724 | $4.55M |
| drop entirely | 0.5655 | 0.8664 | $4.90M |

Recommendation: collapse to the nearest 12. Keeps genuine term-length signal, deletes the
artifact, costs 0.052 headline PR-AUC that was never real. It moves the forecast, so it's
a business decision, not a code decision.

*`days_in_current_stage` (0.206) + `stall_ratio` (0.114) — train/serve skew, needs
snapshots.* Frozen at close on a training row, live on an open deal. Won deals sit at a
median 13 days in their final stage (submit → won is fast), lost deals at 71 — and 80% of
all training wins are in the ≤30-day bucket. A young open deal gets that signal for free.

**The one ask: start snapshotting `dim_opportunity` daily.** Not a daily dataset rebuild —
one cron that appends today's open-deal state to a history file. It is the only fix for
the skew, and it also unlocks stage as a feature, close-date push count,
champion-identified date, honest backtesting, and the head-to-head against the incumbent
that §0 says we can't run.

**It cannot be backfilled.** `fact_opportunity`'s changelog starts 2026-05-01; deals go
back to 2023-01. Every day without it is a day of history that doesn't exist later.

---

## 13. Where each number lives

| number | cell | file |
|---|---|---|
| test metrics (0.630 / 0.879 / 0.067) | 12 | — |
| reliability table | 13 | — |
| overfit vs shift split | 15 | — |
| 26-model comparison | 17 | — |
| the forecast | 23 | `ml_predictions.csv` |
| at-risk list | 26 | — |
| feature set + gate | — | `make_dataset.py` |
| the ACV filter and its reasoning | — | `build_features.py` (`DROP_LOST_WITHOUT_AMOUNT`) |
| model selection, full path | — | `pick_shipping.py` |

**Data handling.** The notebook has **0 saved output cells** and must stay that way — a
saved output embeds the customer rows it printed, and a committed `.ipynb` then leaks them
even though the CSVs are gitignored. Before committing:

```bash
jupyter nbconvert --clear-output --inplace win_probability_colab.ipynb
```

Uploading the CSVs to Colab moves real DevRev CRM data — deal values, account IDs, rep
assignments, named champions — onto Google-hosted infrastructure where notebooks are
shareable by link. That needs data-governance clearance before any wider demo. The repo
stays **private**: this file and the committed markdown carry internal commercial figures.
