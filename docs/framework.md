# Learned Forecasting — Framework & Phase 1 Feasibility

Replacing deterministic forecast (`deal amount × admin-set stage probability`) with a
per-tenant learned, calibrated win probability.

Companion doc: `field_analysis.md` (field-by-field profile, feature set, metrics).
Data checks in this doc run against the DevRev org (`DEV-0`), 2026-08-08.

---

## 1. What's wrong with the current forecast

| Property today | Consequence |
|---|---|
| Stage probability set by hand by an admin | Same constant for every deal in a stage — no differentiation |
| Stage is the only real input | Rep sets it manually; forecast inherits every hygiene gap and sandbagging habit |
| No time dimension | A deal 5 days into "Evaluate" scores identical to one stalled 200 days |
| No deal health | Zero stakeholders, no champion, 3 close-date pushes — all invisible |
| Output = one number | No confidence interval, so no way to express risk |
| No explanation | Rep can't act on it; manager can't challenge it |
| No audit trail | Can't answer "why did the forecast move last week" |

Empirical support from this org: the rep-entered `probability` field is effectively binary — 100 on 203 of 230 won deals, 0 on 4,546 others. It is a post-hoc record of what happened, not a forward-looking estimate. Same for `forecast_category` (`Won` category = won deals). **The org has no working forward-looking probability today.** That's the gap.

## 2. The closed loop

```
Input signals  →  Computation  →  Outputs & next steps  →  Feedback
     ↑                                                        │
     └────────────── retrain / re-tune weights ───────────────┘
```

| Stage | What happens |
|---|---|
| **Input signals** | CRM quantitative fields, historical base rates, (later) transcripts/emails/notes — all normalised into *typed features* |
| **Computation** | ML on quantitative + LLM extraction on qualitative → calibrated win probability, projected close date, expected amount → period revenue **range** |
| **Outputs** | Per-deal score, health band, top 3 drivers with linked evidence, recommended next step targeting that deal's specific risk |
| **Feedback** | Closed outcomes + user corrections retrain the model and re-tune signal weights → accuracy compounds with usage |

Why the loop matters more than the model: a static model decays as pipeline behaviour shifts. The feedback edge is what makes this durable rather than a one-off score.

## 3. Typed-signal contract

Every signal — quantitative or LLM-extracted — carries the same envelope. This is what makes evidence links, audit trail, and weight re-tuning possible:

| Attribute | Meaning | Why required |
|---|---|---|
| `name` | Signal identifier (`champion_identified`) | Stable key for weight tuning |
| `value` | Typed value (bool / number / enum) | Never free text into the model |
| `polarity` | Positive / negative / neutral toward winning | Drives driver explanations |
| `confidence` | 0–1 extraction confidence | Low-confidence signals get down-weighted, not dropped |
| `occurred_at` | When the underlying event happened | **Not** extraction time. Required for point-in-time training (§6) |
| `evidence_ref` | Link to source (CRM field, transcript span, email id) | The audit trail. No evidence ref = signal not shown to users |
| `source` | `crm` / `transcript` / `email` / `note` / `derived` | Lets you measure which source actually adds lift |

Rule: a signal with no `evidence_ref` can feed the model but must never appear in a driver explanation. Unexplainable drivers are how forecasting tools lose trust.

---

## 4. Phase 1 feasibility — verified against real data

### 4a. Quantitative CRM signals

| Signal | Source | Available? | Notes |
|---|---|---|---|
| Deal amount | `annual_contract_value` ∪ `value` | ⚠️ **31% fill** | All USD, no FX needed. Impute + flag (see `field_analysis.md` §6) |
| Days in stage | `tnt__days_in_current_stage` | ✅ 91.2% | Also derivable from revision history |
| Days in sales cycle | `tnt__days_in_sales_cycle` | ✅ 91.2% | |
| **Close-date pushes** | Revision history | ✅ **1,321 deals have >1 distinct `target_close_date`**, avg 2.6, max 10 | Not a stored field — must be mined (§5) |
| **Amount changes** | Revision history | ✅ **690 deals have >1 distinct ACV**, max 10 revisions | Must be mined |
| Stakeholders | `len(contact_ids)` | ✅ 91.7% | |
| Activity recency | `modified_date` vs now | ✅ 100% | Avg 44 edits/deal, max 3,083 |
| Rep win rate | Derived from closed history | ✅ 247 reps | Leave-one-out to avoid leakage |
| Stage | `stage_json.ordinal` | ✅ 100% | Strongest single signal |

**All nine are achievable.** Two require mining the revision history rather than reading a column.

### 4b. Qualitative signals — ⚠️ blocked in this org

Tables exist and are queryable:

| Table | Rows | Linked to opportunities |
|---|---|---|
| `fact_meeting` | 3,607,528 | 1,720 meetings across **1,348 deals** |
| `fact_engagement` | 2,217,876 | 234 across **202 deals** |
| `fact_conversation` | 221,810 | support conversations, has `sentiment_json` |
| `fact_account`, `dim_account` | 1.66M / 211K | for account-level base rates |
| `fact_survey_response` | 9,984 | |
| `fact_timeline_entry`, `fact_email`, `fact_work`, `fact_rev_user` | — | **404 — not present** |

**The blocker:** of 3.6M meetings, 153,956 have a `transcript_id`. But meetings *linked to an opportunity* with a transcript:

```
opps_w_transcript = 2
```

**Two deals.** Not 2%. Two.

So Phase 1's qualitative half cannot be trained or validated on this org's data. `fact_email` doesn't exist at all. Notes exist as `tnt__*` free-text MEDDPICC fields (14–34% fill) — usable, but as a completeness count, not as extraction targets.

Options, pick before committing Phase 1 scope:

| Option | Trade-off |
|---|---|
| **A. Ship quantitative-only Phase 1**, qualitative in Phase 2 | Honest, deliverable now. Recommended |
| **B. Find a design-partner org with transcript coverage** | Validates the full vision, but needs a tenant that actually records calls against deals |
| **C. Build extraction against `fact_conversation`** (221K support convos, has `sentiment_json`) | Real text at volume — but it's *support* sentiment, not sales-call sentiment. Different signal, weaker link to win probability |
| **D. Use MEDDPICC free-text fields as the qualitative surface** | Available today (`tnt__why_now`, `tnt__identified_pain`, `tnt__decision_criteria`, …). Rep-written, so measures rep diligence as much as deal health |

Recommendation: **A now, D as a cheap add, B in parallel to de-risk Phase 2.** Don't design the typed-signal contract around data that isn't there — but *do* build the contract now so extraction plugs in without rework.

### 4c. Per-tenant model — ⚠️ label-count constrained

Vision: model trained on each org's own closed-won/lost history, base rates by account, industry, segment, rep.

This org's reality: **4,915 closed deals but only 230 wins (4.7%).**

| Base rate | Groups | Wins available | Verdict |
|---|---|---|---|
| By segment | 5 (Enterprise 2261 / Mid-Market 1163 / SMB 986 / Start-Up 857 / 2K 306) | ~46 avg | Workable with smoothing |
| By rep | 247 | **~1 avg** | Not directly estimable — needs hierarchical shrinkage toward global rate |
| By account | 5,019 accounts / 6,900 deals | ~0.05 | Not estimable. Use `account_prior_deals` count instead |
| By industry | Not a field on `fact_opportunity` | — | Must join `dim_account`. Unverified |
| By geo | 4 (AMER 3046 / APJ 2841 / EMEA 845) | ~57 avg | Workable |

**Implication for "per-tenant":** a fully independent per-tenant model needs on the order of a few hundred wins. Below that, per-tenant training overfits and will look worse than the hand-set constant it replaces — the fastest way to kill adoption.

Design for this: **hierarchical / partial pooling.** Global prior across tenants, shrunk toward tenant-specific estimates in proportion to that tenant's label count. A tenant with 30 wins mostly inherits global behaviour; one with 2,000 wins is effectively fully self-trained. Same treatment for rep and account base rates within a tenant. This also solves cold-start — a new tenant gets a usable score on day one instead of waiting a year.

**What "learns different business patterns" requires — and what v0 can't do.** Verified
2026-08-10: `dim_cache.parquet` has `dev_oid` nunique = **1** (DEV-0, 12,963 deals).
Cross-business generalization needs multiple businesses in *training*; 2023-DevRev vs
2025-DevRev is one business at two times, and every difference between them is confounded
with time (product, market, team, lead-gen all changed). No feature can represent "which
business am I in", so the pattern has no column to live in. Split the goal in two:

| | Transfers? |
|---|---|
| **Ranking** — "does a high `stall_ratio` mean trouble anywhere?" | Likely yes. Directional, probably survives across orgs |
| **Base rate** — "is 30% the right number *here*?" | No. A 40%-win org and a 3%-win org need different absolute probabilities from identical features — and this is the half that ships, since probability × amount = revenue |

The design that gets both, on top of the pooling above:
1. **Global ranking model** on pooled deals from many orgs, with **tenant-relative**
   features — `days_in_stage` as a percentile within *that org's* stage distribution, not
   raw days. Absolute days let a fast-cycle org and a slow-cycle org poison each other.
2. **Per-tenant calibration layer** — a thin Platt fit on that org's own closed deals,
   shrunk toward global by label count. This is what makes the absolute number right.
3. **Tenant meta-features** as inputs: org base rate, median cycle length, pipeline
   volume. Finally gives "which business" a column to live in.

Recency drift is then just another shrinkage axis, same machinery.

**Build order:** v0 targets DEV-0, then scales. Worth building the tenant-relative
features and the separate calibration layer *now*, even with one tenant, so adding org #2
is a data change rather than a rewrite. Before committing to the architecture, one query
settles it: pull `dev_oid != 'DEV-0'` from the same source and count wins per org. Two or
three orgs at ≥100 wins each means transfer can be *tested* rather than assumed.

Also required per tenant: **each org has its own stage schema.** This org alone carries two (numbered `1-profile`…`9-closed_lost`, plus legacy SFDC `Prospecting`/`Needs Analysis`/…) and 145 custom fields with tenant-specific names. Feature extraction must be schema-driven, not hardcoded to `tnt__*` keys.

### 4d. Per-deal output

| Output | Feasible | Notes |
|---|---|---|
| Calibrated win probability | ✅ | Calibration is the hard part, not classification. Isotonic/Platt on a held-out slice; monitor with reliability curves |
| Health Green/Yellow/Red/**Unknown** | ✅ | The `Unknown` band is essential here — 69% of deals lack amount, 19% lack segment. Better to say Unknown than to score confidently on nothing. Define thresholds on *calibrated* probability, and require minimum signal coverage to leave Unknown |
| Top 3 drivers with evidence | ✅ | SHAP on the quantitative model gives per-deal attribution. Only surface drivers that carry an `evidence_ref` |
| Projected close date | ⚠️ | Separate regression on cycle length. `target_close_date` 86.5% filled, `actual_close_date` 71% — trainable, but treat as a distinct model with its own metric (MAE in days) |
| Expected amount | ⚠️ | 31% amount fill undermines this. Expected amount on an imputed amount is a guess dressed as a number — must show as a range, and flag imputed deals |
| Period revenue range | ✅ | Sum of calibrated probabilities gives the point estimate; the range needs prediction intervals, not just Σp. Requires calibration to actually hold |

## 5. Revision history — the underused asset

`fact_opportunity` has 303,415 rows for 6,900 deals (~44 revisions each). Currently deduped to latest revision for training (correctly — see `field_analysis.md` §0). But the discarded 296K rows are exactly the time-and-health dimension the current forecast lacks:

| Derivable feature | Verified availability |
|---|---|
| Close-date push count / total slip days | 1,321 deals with >1 target date, avg 2.6, max 10 |
| Amount revision count / direction | 690 deals with >1 distinct ACV, max 10 |
| Stage transition sequence, dwell time per stage, **backward moves** | 2,822 deals with >1 stage, avg 2.28 |
| Edit velocity / recency curve | avg 44 edits per deal |
| **Point-in-time feature snapshots** | Any deal state at any past date is reconstructable |

That last row is the important one. It fixes the 230-label problem two ways:

1. **Training rows multiply.** Instead of one row per closed deal, generate one row per (deal, observation date) — "what did this deal look like 30/60/90 days before close?" Same 230 outcomes, many more training examples, and the model learns to score deals *mid-flight*, which is the actual production use case.
2. **Honest backtesting.** Reconstruct pipeline as of a past quarter-end, predict, compare to what actually closed. That's the only credible way to show a manager this beats the current constant.

Caveat: point-in-time reconstruction is where leakage sneaks in. Every feature at observation date `T` must use only revisions with `object_version` ≤ the version live at `T`. This is why `occurred_at` is mandatory in the signal contract.

## 6. Leakage discipline

Six fields in this org are filled *because* the deal closed (full table in `field_analysis.md` §1):

| Field | Fill on won | Fill on lost |
|---|---|---|
| `tnt__win_notes` | 84.3% | 0.0% |
| `tnt__closed_lost_reasons` | 1.3% | 93.1% |
| `tnt__ramped_arr` | 91.3% | 25.7% |
| `tnt__agentic_score` | 23.5% | 70.8% |
| `probability` | 100 on 203/230 wins | 0 on 4,546 |
| `tnt__next_steps` | 87.8% | 41.9% |

This is a **framework requirement, not a one-off cleanup**. A per-tenant system ingesting arbitrary custom fields will hit tenant-specific leakage every time. Build it in:

- Automated leak scan per tenant: flag any feature whose fill-rate or value distribution diverges sharply by outcome, before training.
- Point-in-time rule: a signal may only enter a training row if `occurred_at` precedes the observation date.
- Sanity gate: any model scoring above ~0.95 ROC-AUC on this problem is leaking. Treat high accuracy as a bug report.

`tnt__agentic_score` deserves specific mention — it's AI-generated, 71% filled on losses vs 24% on wins. Training on it means predicting an AI's output with an AI. Circular, and it will silently inherit that model's biases.

## 7. Metrics & baselines

Full detail in `field_analysis.md` §7. The framework-level points:

| Metric | Why it's the right one here |
|---|---|
| **PR-AUC** (target > 0.25, baseline 0.047) | Primary. Accuracy is meaningless at 4.7% — "always lose" scores 95.3% |
| **Brier score** | Calibration is the product. A well-ranked but miscalibrated model produces a wrong revenue number |
| **Reliability curve**, 10 bins, ±10% | "Says 30% → ~30% close." This is what makes the roll-up range defensible |
| **Recall @ top-20%** (> 50%) | How a manager actually uses it |
| **Beat rep `probability`** on Brier + PR-AUC | The honest baseline |
| **Beat the current forecast** — stage-constant × amount | **The baseline that decides ship/no-ship.** This is the incumbent. Must be reported every retrain, per tenant |
| **Beat stage-only logistic regression** | Guards against a 23-feature model that only learned stage |

Validation splits by `created_date`, never randomly — random splits leak future into past through `rep_win_rate` and account history.

**Feedback-loop metrics** (the part vision-docs usually omit): track user-correction rate per driver type, and whether accuracy actually improves per retrain. If corrections don't measurably raise accuracy, the loop is decorative and should be cut.

## 8. Recommended phasing

**Phase 1 (deliverable on current data)**
- Quantitative CRM signals — all 9 verified available
- Revision-history features: close-date pushes, amount revisions, stage transitions, backward moves
- Point-in-time training rows to expand 230 labels into a usable set
- Hierarchical base rates (segment, geo, rep-with-shrinkage)
- Calibrated probability + Green/Yellow/Red/**Unknown** + top-3 SHAP drivers with CRM field evidence links
- Typed-signal contract implemented in full, even though only `source=crm` and `derived` populate it
- Backtest against stage-constant baseline on a past quarter

**Phase 2 (needs data that doesn't exist yet here)**
- Transcript/email extraction — gated on a tenant with real coverage (currently 2 deals)
- MEDDPICC free-text extraction as the interim qualitative surface
- Projected close date and expected amount as separate models with their own metrics
- Recommended-next-step generation
- Correction feedback → weight re-tuning

**Sequencing note:** ship the loop's *feedback plumbing* in Phase 1 even with only quantitative signals. Retrofitting an audit trail after users are already looking at scores is far more expensive than building it now.

## 9. Open questions

Data-answerable, not yet checked:
- Does `dim_account` carry industry? Needed for industry base rates.
- Is `fact_conversation` sentiment linkable to opportunities via `account_id`? Would give a real text signal today.
- Which other orgs have meeting transcripts linked to opportunities? Determines whether Phase 2 is testable at all.

Needs sales/product input:
- What does `state = in_progress` (98 deals) mean vs `open`? In or out of training?
- Is `tnt__segment = "2K"` (306 deals) real, or a data-entry artifact?
- Legacy SFDC stage names (`Prospecting`, `Needs Analysis`, …) — dead pipeline or live?
- Health band thresholds: who owns the Green/Yellow/Red cutoffs, and what does a rep do differently at each?
- ACV unpopulated on ~69% of deals is a process gap worth escalating regardless of this project.
