# Field Analysis & Feature Set — forecast_v0

Deal win-probability model. Source: `devrev.oasis.data.query` → `devrev.fact_opportunity`.
Profiled 2026-08-08 on all 6,900 opportunities (latest revision each).

---

## 0. Dataset correction (read this first)

`fact_opportunity` is a **change-log**, not a deal list. One row per edit.

| Check | Result |
|---|---|
| `count(*)` | 303,415 |
| `count(distinct id)` | 6,900 |
| `count(distinct display_id)` | 6,900 |
| `count(distinct (id, object_version))` | 303,415 — exactly row count, so `(id, object_version)` is the PK |
| `count(distinct dev_oid)` | 1 (single org — duplication is not multi-tenant) |
| Worst case: `opportunity/11154` | 3,083 rows / 3,083 distinct `object_version` / 3,083 distinct `modified_date` |

6,900 real deals × ~44 saved revisions each = 303,415 rows.

Fixed in `extract_pipeline.py:32` — take newest revision per deal:

```sql
SELECT * FROM (
SELECT row_number() OVER (PARTITION BY id ORDER BY object_version DESC) AS rn, ...
FROM devrev.fact_opportunity WHERE is_deleted = false
) WHERE rn = 1 ORDER BY id
```

**True population:**

| | Count |
|---|---|
| Total deals | 6,900 |
| Closed (labelled, trainable) | 4,915 |
| — won (`8-closed_won` 212 + `Closed Won` 18) | **230** |
| — lost (`9-closed_lost`) | 4,685 |
| Open (inference targets) | 1,887 |
| `in_progress` (semantics unknown) | 98 |
| **Win rate** | **4.7%** |

Consequences:
- Every figure in `amount_gap_decision.md` is void (299,943 opps / 69% missing / 22.8% win rate) — inflated ~44× and weighted by how often each deal was edited.
- Earlier "16.3% win rate" was also wrong — that was the amount-populated subset only.
- **230 positives is the real constraint on this project.** Small-data problem, not a modeling problem.
- Upside: the 303k revisions are a genuine asset later — stage transitions and close-date pushes are recoverable from them (§7).

---

## 1. Label leakage — 10 fields must be excluded

Fill rate differs drastically by outcome, i.e. these are filled *because* the deal closed:

| Field | Fill on WON | Fill on LOST |
|---|---|---|
| `tnt__closed_lost_reasons` | 1.3% | 93.1% |
| `tnt__win_notes` | 84.3% | 0.0% |
| `tnt__ramped_arr` | 91.3% | 25.7% |
| `tnt__agentic_score` | 23.5% | 70.8% |
| `tnt__next_steps` | 87.8% | 41.9% |
| `probability` | 100 on 203 of 230 wins | 0 on 4,546 |

Also leaking: `forecast_category` (`Won` category = the label), `tnt__probability_by_forecast` (100 = won), `tnt__weighted_pipeline_*` (contains probability), `actual_close_date`, all `ctype__*` (includes `ctype__iswon_*`).

Any of these in the feature set yields ~0.99 AUC that means nothing in production.

## 2. Amount / currency

| Field | Populated | Currency |
|---|---|---|
| `amount` (stock column) | **0%** | — |
| `value.amount` | 23.3% | USD only |
| `annual_contract_value.amount` | 30.7% | USD only |
| `budget.amount` | 0.4% | USD only |
| `annual_recurring_revenue`, `customer_budget` | 0% | — |

**No FX handling needed** — every populated row is USD.

Use `annual_contract_value` as primary, coalesce `value` as fallback → ~31% coverage. Stored as strings inside a JSON struct (`{"amount":"30000","currency":"USD"}`) — parse with `pd.to_numeric(..., errors="coerce")`.

---

## 3. Stock fields

| Field | Stores | Type | Use? | Handling |
|---|---|---|---|---|
| `id` / `display_id` | Deal key (`OPP-10000`) | ID | Key only | Join key, never a feature |
| `object_version` | Revision no., 1–3116 | Int | Dedup + derive | `rn=1` for latest; high version = heavily-edited = engagement proxy |
| `state` | open 1887 / closed 4915 / in_progress 98 | Cat (3) | Split, not feature | closed=train, open=inference. Clarify `in_progress` before using |
| `stage_json.name` | 27 values: numbered (`1-profile`…`9-closed_lost`) + legacy SFDC (`Prospecting`, `Needs Analysis`…) | Cat | **Yes — top signal** | Ordinal-encode numbered; drop ~14 legacy rows. Closed stages define the label — exclude for open deals |
| `stage_json.ordinal` | Funnel position | Ordinal | **Yes** | Direct numeric |
| `annual_contract_value` | ACV, USD | Quant | **Yes — primary amount** | Best fill (30.7%). Log-transform |
| `value.amount` | Deal size, USD | Quant | **Yes — fallback** | 23.3%. Coalesce into ACV |
| `amount`, `annual_recurring_revenue`, `customer_budget`, `external_ref`, `reported_by_id`, `last_internal_comment_*` | — | — | **Drop** | 0% populated |
| `budget` | Buyer budget | Quant | **Drop** | 0.4% fill |
| `probability` | Rep-entered %, 10 values | Quant | **Drop — leakage** | Use as *baseline to beat* |
| `forecast_category` | Omitted 4661 / Pipeline 1875 / Won 230 / Upside 101 / Strong Upside 21 / Commit 12 | Cat (6) | **Drop — leakage** | `Won` IS the label |
| `priority` | 3 on 6,694 of 6,900 | Cat (4) | **Drop** | No variance (97% one value) |
| `created_date` | Epoch ms | Date | **Yes, derived** | → deal age, quarter seasonality. Never raw |
| `target_close_date` | 86.5% fill | Date | **Yes, derived** | → days-to-close, push count vs history |
| `actual_close_date` | 71% fill | Date | **Label-time only** | Leakage as a feature |
| `modified_date` | Last touch | Date | **Yes, derived** | → staleness (days since touch). Strong signal |
| `owned_by_ids[0]` | Rep (~247) | Cat | **Yes, encoded** | Leave-one-out win rate. High cardinality → never one-hot |
| `account_id` | 5,019 accounts / 6,900 deals | Cat | **Derived only** | → prior deal count, prior win rate |
| `contact_ids` | Stakeholder list, 91.7% | Quant | **Yes** | `len()`; null → 0 |
| `created_by_id` (247) / `modified_by_id` (128) | Actor | Cat | Maybe | svcacc vs human = bulk-import vs real deal; data-quality flag |
| `title` | Free text, 6,631 uniq | Free text | Defer to v1 | Keyword flags (`Renewal`, `Expansion`, `POC`) |
| `tags_json` (560) / `links_json` (1021) / `reference_ids` (4.9%) | Relations | Sparse | Defer | `len()` only |
| `apps` | 12.8%, `bulk_work_item_snapin` | Cat | **Yes as flag** | Bulk-imported deals behave differently — control variable |
| `dev_oid`, `object_type`, `work_type`, `is_deleted`, `rn`, `subtype`, `staged_info`, `devrev_copy`, `event_context`, `shared_with`, `reactions_json`, `sync_metadata`, `unread_notifications_metadata`, `external_source_data`, `*schema_fragment*`, `*deprecated*`, `system_metadata` | Plumbing | — | **Drop** | Single-valued or internal |

## 4. Custom fields (145 keys total — the ones that matter)

| Field | Stores | Fill | Type | Use? | Handling |
|---|---|---|---|---|---|
| `tnt__opportunity_type` | New 6271 / Renewal 372 / Upsell 254 / Amendment 3 | 100% | Cat (4) | **Yes — strong** | One-hot. Renewals win differently |
| `tnt__source` | Outbound 4189 / Inbound 1490 / Events 784 / Partner 423 | 99.9% | Cat (6) | **Yes** | One-hot. Classic win-rate driver |
| `tnt__geo` | AMER 3046 / APJ 2841 / EMEA 845 | 97.6% | Cat (4) | **Yes** | One-hot |
| `tnt__segment` | Enterprise 2261 / Mid-Market 1163 / SMB 986 / Start-Up 857 / 2K 306 | 80.8% | Cat (5) | **Yes** | `unknown` for missing 19%. Confirm what `2K` means |
| `tnt__contract_months` | 12 typical, 36 uniq | 99.8% | Quant | **Yes** | Median-fill |
| `tnt__days_in_current_stage` | 418 uniq | 91.2% | Quant | **Yes — strong** | Stall detector. Median-fill **by stage**, not 0 |
| `tnt__days_in_sales_cycle` | 486 uniq | 91.2% | Quant | **Yes** | Median-fill by stage |
| `tnt__region` (17) | Geo detail | 95.6% | Cat | **Yes** | One-hot |
| `tnt__sub_region` (43) | Finer geo | 92.9% | Cat | **No** | Too granular for 230 positives |
| `tnt__poc` | True 320 | 100% | Bool | **Yes** | POC ran = real evaluation |
| `tnt__technical_voc` | True 192 | 100% | Bool | **Drop — train/serve gap** | 8.1% of closed deals carry it vs **0.1% of open** (896 vs 2). Field stopped being maintained; AUC 0.5176 |
| `tnt__nda` | True 93 | 100% | Bool | **Drop — overfits** | 40.6% win rate on the 138 closed deals with it, but only 56 wins behind it. Removing it *improves* PR-AUC 0.5631 → 0.5781 over 5 seeds |
| `tnt__bill_through` | DevRev 6880 / Reseller 13 / AWS 4 | 100% | Cat | **Drop** | No variance |
| `tnt__partner_sourced` | True on 11 | 100% | Bool | **Drop** | 11 positives. Currently in `FEATURES` — remove |
| `tnt__fiscal_quarter_created` (15) / `_close` (29) | Fiscal period | 99.4 / 86.2% | Cat | **Derived** | → quarter-end flag; deals cluster at boundaries |
| `tnt__champion` | Champion contact | 34.7% | ID | **Yes as bool** | `has_champion` — MEDDIC signal. Never the ID |
| `tnt__economic_buyer` | EB contact | 11.3% | ID | **Yes as bool** | `has_economic_buyer` |
| `tnt__sales_engineer` / `_sales_development_representative` / `_pbm` / `_customer_success` | Team assigned | 24.8 / 38.1 / 6.6 / 4.4% | ID | **Yes as bool** | `has_se`, `has_sdr` = resourcing signal |
| `tnt__competition` | List, e.g. `['Not Applicable']` | 34.6% | Multi | **Yes as bool + count** | `has_competition`, `n_competitors` |
| `tnt__use_case` / `_playbook` / `_plans` / `_primary_products` | Multi-select lists | 17.4 / 21.5 / 3.7 / 2.4% | Multi | **Count only** | Too sparse to one-hot |
| `tnt__why_anything` / `_why_now` / `_why_devrev` / `_identified_pain` / `_decision_criteria` / `_decision_process` / `_metrics` | MEDDPICC free text | 34.4 / 17 / 14.9 / 18.8 / 17.8 / 7.8 / 6.9% | Free text | **Yes — as count** | Don't embed. `meddpicc_completeness = count(non-null of these 7)`, one integer. Rep-diligence proxy; most likely genuinely predictive family |
| `tnt__next_steps` / `_next_step_due_date` | Next action | 44 / 42.5% | Free / Date | **Bool only, cautiously** | 88% won vs 42% lost = partial leakage. `has_next_step_overdue` safer |
| `tnt__weighted_pipeline_stage` / `_forecast_category` | amount × probability | 94.3 / 93.2% | Quant | **Drop — leakage** | Contains probability |
| `tnt__probability_by_forecast` | 0 / 10 / 100 | 93.3% | Quant | **Drop — leakage** | 100 = won |
| `tnt__ramped_arr` | Post-close ARR | 32.6% | Quant | **Drop — leakage** | 91% won vs 26% lost |
| `tnt__agentic_score` / `_primary_blocker` / `_suggested_next_steps` | AI-generated | 50.6 / 4.1 / 4.0% | Mixed | **Drop — leakage** | 71% lost vs 24% won; also circular (AI output predicting AI target) |
| `tnt__closed_lost_reasons` / `_details` | Loss reason | 63.4 / 64.7% | Multi / Free | **Drop — pure leakage** | Gold for loss analysis, poison as feature |
| `tnt__win_notes` | Win writeup | 2.9% | Free text | **Drop — pure leakage** | 84% won, 0% lost |
| `tnt__contract_start/end/signed_date`, `_number_of_*_licenses`, `_commission*`, `_billing_address`, `_legal*`, `_infosec`, `_base_arr`, `_implementation_cost`, `_opt_out*`, `_poc_start/end/result/notes`, `_registration_*`, `_manager_*`, `_co_sell_*`, `_bill_through_*`, `_partner_*`, `_aai_*`, `_advisor_contact`, `_selected_vendor`, `_sell_to`, `_visible_to`, `_segment_override`, `_incumbent_*`, `_current_techstack_*`, `_required_capability`, `_solutions_architects_sa`, `_additional_ses`, `_marketing_development_representative` | Post-sale / ops / partner detail | **<7%** | Mixed | **Drop** | Below 7% fill against 230 positives; mostly post-close artifacts |
| `ctype__*` (23 fields) | Salesforce import residue — fiscal year 2015, competitors "John Deere"/"Honda" | **0.0–0.4%** (31 rows) | Mixed | **Drop all** | Demo/sandbox data from SFDC migration. `ctype__iswon_*` is literally the label |
| `app_bulk_work_item_snapin__unique_id_opportunity` | Bulk-import row ref | 12.8% | ID | Drop as ID, keep as flag | Same signal as `apps` |

Data-hygiene note: `tnt__registration_status` has `Approved` / `APPROVED` / `approved` as three values. Normalise case on any categorical before encoding.

---

## 5. Recommended v0 feature set — 23 features

**Quantitative (7)**
`log_acv` (ACV coalesced with value), `stage_ordinal`, `days_in_current_stage`, `days_in_sales_cycle`, `contract_months`, `num_stakeholders`, `days_since_modified`

**Derived (5)**
`rep_win_rate` (leave-one-out), `account_prior_deals`, `meddpicc_completeness` (0–7), `n_competitors`, `edit_count` (from `object_version`)

**Boolean (7)**
`has_champion`, `poc`

Superseded — see `dataset_sample.md` for the shipping 11-field set. `technical_voc`
dropped (0.1% fill on open deals); `nda` dropped (overfits 56 wins);
`has_economic_buyer` / `has_sdr` / `is_bulk_imported` never made it in; `has_se`
held for point-in-time.

**Categorical, one-hot (4)**
`opportunity_type`, `source`, `geo`, `segment`

**Excluded for leakage (10)**
`probability`, `forecast_category`, `tnt__probability_by_forecast`, `tnt__weighted_pipeline_*`, `tnt__ramped_arr`, `tnt__agentic_*`, `tnt__closed_lost_*`, `tnt__win_notes`, `actual_close_date`, all `ctype__*`

## 6. Missing-data policy

| Situation | Policy | Why |
|---|---|---|
| ACV / value missing (~69%) | **Impute median by segment × opportunity_type, plus `amount_was_imputed` flag** | Dropping cuts closed 4,915 → 1,359 and biases toward wins. The flag lets the model learn "no amount recorded" as its own signal — which it is |
| `days_in_*` missing (~9%) | Median **by stage**, not 0 | Current `fillna(0)` fabricates "brand-new deal" |
| Categorical missing | Literal `"unknown"` category | Missingness is informative here |
| MEDDPICC text missing | Counts as 0 in completeness | That's the feature's purpose |
| Boolean-from-ID missing | `False` | Absence = not assigned |
| Categorical case variants | Lowercase-normalise | `Approved`/`APPROVED`/`approved` |
| Legacy SFDC stage names (~14 rows) + `ctype__` rows (31) | **Drop rows** | Different pipeline schema, unmappable |

## 7. Metrics

Class balance 4.7% — **accuracy is useless** (always-predict-lose scores 95.3%).

| Metric | Target | Why |
|---|---|---|
| **PR-AUC** | > 0.25 | Primary. Baseline 0.047. Report ROC-AUC, don't optimise it — flatters imbalanced data |
| **Brier score** | < rep baseline | Forecasting needs *calibrated* probabilities, not just ranking. Sum of predictions ≈ expected deal count |
| **Calibration curve** | 10 bins, within ±10% | If model says 30%, ~30% must close. This is what makes the aggregate forecast trustworthy |
| **Recall @ top-20% ranked** | > 50% | Operational reading: of deals flagged likely, how many landed |
| **Beat rep `probability`** | Must win on Brier + PR-AUC | Honest baseline — reps already assign probability. Lose here and v0 has no reason to ship |
| **Beat stage-only model** | Must win | Logistic regression on `stage_ordinal` alone. Guards against a 23-feature model that only learned stage |

**Validation: split by `created_date`, never random.** Train older, test newer. Random split leaks future into past via `rep_win_rate` and account history. Use 5-fold time-series CV and report the spread — a single split gives ~46 test positives, which is noise.

**Caveat:** 230 wins across 23 features is thin. Expect regularised logistic regression or shallow gradient boosting to beat anything deeper. If PR-AUC < ~0.15, the fix is more labelled data (mine the 303k revisions for stage-transition sequences), not more features.

## 8. Next steps

1. Rewrite `extract_pipeline.py` to this feature set + imputation policy — unblocks training
2. Mine the history table: 303k revisions → stage-transition counts, close-date pushes, amount revisions. Best answer to thin labels
3. Verify leakage empirically — train once *with* `probability` to demonstrate the ~0.99 AUC, so nobody re-adds it
4. Flag to CRM/data owners: ACV unpopulated on ~69% of deals (process gap, worth fixing regardless of this project)

**Open questions for sales — can't be answered from data:**
- What does `state = in_progress` (98 deals) mean vs `open`? Include in training or not?
- Is `tnt__segment = "2K"` (306 deals) a real segment or a data-entry artifact?
- Legacy SFDC stage names (`Prospecting`, `Needs Analysis`, …) — dead pipeline or still live?
