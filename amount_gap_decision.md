# Missing `amount` — Decision Table

## The problem, in numbers

| Metric | Value |
|---|---|
| Total opportunities (all states) | 299,943 |
| Amount populated (all states) | 93,198 (31%) |
| Amount missing (all states) | 206,745 (69%) |
| Closed deals only | 25,856 |
| Closed deals with amount | 10,976 (42%) |
| Closed deals missing amount | 14,880 (58%) |
| Win rate — full closed population | 22.8% (5,895 won / 19,961 lost) |
| Win rate — amount-populated subset only | 53.5% |

**Key finding**: deals missing `amount` skew heavily toward losses. Dropping them (naive approach) doubles the apparent win rate — training on that subset would make the model systematically overconfident in production.

## Root cause — ruled out stage timing, confirmed it's loss-specific

Hypothesis considered: maybe deals die early (before a rep ever sizes them), so missing amount is just a side effect of early-stage abandonment, not a real bias.

Tested directly — split closed deals by `has_amount`, checked stage_ordinal and win rate within each stage:

| stage_name | has_amount | count | win rate |
|---|---|---|---|
| 8-closed_won | False | 30 | 100% |v
| 8-closed_won | True | 5,869 | 100% |
| 9-closed_lost | False | 14,862 | 0% |
| 9-closed_lost | True | 5,053 | 0% |

**Result: hypothesis rejected.** `stage_ordinal` medians are nearly identical between `has_amount=True/False` groups (~7200 either way) — missing-amount deals are NOT dying earlier in the funnel. Instead, missing amount is concentrated almost entirely in `9-closed_lost` (14,862 of 14,892 total missing-amount rows), regardless of stage. Most plausible explanation: **reps stop maintaining deal amount once they've mentally written off a deal** — a data-hygiene/process gap, not a timing artifact.

Side finding from this check: label logic had a bug — 54 rows used stage name `"Closed Won"` (capitalized variant) instead of `"8-closed_won"`, silently mislabeled as losses. Fixed in the pipeline (`is_won` now matches both variants).

## Options

| Option | How it works | Pros | Cons | Effort |
|---|---|---|---|---|
| **A. Drop rows missing amount** (current v0 default) | Train only on the 42% with amount | Simplest, no new logic | Confirmed selection bias — model overestimates win probability | None (already built) |
| **B. Impute missing amount** | Fill nulls with median amount by segment/stage/region | Keeps full dataset, removes selection bias | Imputed values are guesses — adds noise; risk if imputation logic is wrong | Low (1 day) |
| **C. Drop `amount` as a required model input** | Predict `is_won` using only stage, days-in-stage, activity, rep rate — amount used only where present, for aggregation | No bias from dropping rows; model works on 100% of deals | Loses a genuinely useful signal (deal size correlates with win behavior in most orgs) | Low (schema/feature change only) |
| **D. Escalate — fix data capture at source** | Flag to CRM/data owners that amount isn't filled on ~2/3 of deals, especially losses | Fixes root cause, benefits all future work | Slow, doesn't unblock v0 timeline, needs org buy-in | Not in our control |

## Recommendation

**B + D together**: impute for v0 to unblock training now (median by segment+stage, defensible and reversible), while separately flagging D to whoever owns CRM data quality — this is a process gap (reps not filling amount, especially on deals they're about to lose) worth fixing regardless of the ML project.

**Avoid A as-is** — it's already proven biased and shipping a model trained on it risks eroding trust in v0 the moment someone compares it against known deals.

## What this doesn't block
Whichever option is picked, `is_won` prediction can proceed — this only affects how amount enters the feature set and how confidently we produce the amount-weighted aggregate forecast number.
