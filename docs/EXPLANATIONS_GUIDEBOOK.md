# Deal Explanations Guidebook

How to interpret AI-generated deal rankings and feature explanations.

---

## What You're Looking At

Two things per deal:

1. **Win Probability** — model's odds this deal closes (0–100%)
2. **Explanation** — why that probability, plus 3 key factors driving it

Example row:
```
display_id: OPP-13096
win_probability: 76%
signal_strength: high
top_factors: meetings_first_30d (+0.18), days_in_sales_cycle (+0.12), has_champion (+0.09)
explanation: "Champion actively pushing contract. 8 meetings in first month, deal moving fast. Close within 2 weeks."
```

---

## Reading the Factors

Each factor shows:

| Part | Means |
|------|-------|
| **Factor name** | What we measured (meetings, days, champion, etc) |
| **Value** | What we saw (8 meetings, 45 days, yes/no) |
| **SHAP score** | How much it moved the needle (+0.18 = strong positive) |
| **↑ / ↓** | Direction (↑ = helps close, ↓ = risky) |

**Positive factors** (+) = deal looks healthier
**Negative factors** (-) = red flags

---

## Signal Strength Buckets

| Bucket | What It Means | What To Do |
|--------|-------------|-----------|
| **high** | >75% confidence, deal shows real momentum | Prioritize. Close this week if possible. |
| **medium** | 50–75% confidence, some activity but early | Track closely. Nurture actively. |
| **low** | 25–50% confidence, minimal activity | Develop normally. No rush. |
| **none** | <25% confidence, too young to assess | Background. Focus on high/medium first. |

---

## Three Deal Types

### Type A: "Close This Now"

```
Win prob: 82%, Signal: high
Top factors:
  ↑ meetings_per_month: 2.5 (strong engagement)
  ↑ days_since_latest_meeting: 2 (just met today)
  ↑ has_champion: yes (someone owns this)
Explanation: "Active negotiation stage. Champion drove 3 meetings this month. 
Latest touchpoint 2 days ago. Ready for close or final objection handling."
```

**Action:** Call champion, ask for close date. Remove blockers. Don't let it sit.

---

### Type B: "Keep Warm"

```
Win prob: 58%, Signal: medium
Top factors:
  ↑ contract_months: 12 (long deal)
  ↓ days_in_current_stage: 67 (stuck in proposal)
  ↑ num_stakeholders: 4 (multi-stakeholder buy)
Explanation: "Proposal stage taking longer than normal (67 days). Multiple buyers 
need alignment. No recent activity. Recommend stakeholder call to unblock."
```

**Action:** Identify blocker (procurement? legal?). Get champion to drive next step. Set calendar reminder 1 week.

---

### Type C: "Not Ready Yet"

```
Win prob: 18%, Signal: low
Top factors:
  ↓ days_in_sales_cycle: 8 (brand new)
  ↓ meeting_quiet_frac: 0.95 (barely met them)
  ↓ number_of_meetings: 1 (one kickoff, nothing since)
Explanation: "Early stage, not enough signal yet. One kickoff meeting but no 
follow-up. Let mature for 2 weeks, revisit after next business review."
```

**Action:** Schedule their business review, but don't push close. Too early.

---

## Interpreting SHAP Scores

Scale:
- **±0.30+** = Major factor (deal-maker or deal-breaker)
- **±0.10–0.30** = Meaningful (contributing to final prob)
- **±0.01–0.10** = Minor (edge case, but real)

Example interpretation:
```
meetings_first_30d: +0.18
→ "Having 8 early meetings increased close odds by ~18%. Strong early signal."

days_in_current_stage: -0.22
→ "Being stuck 67 days in proposal decreased close odds by ~22%. Major red flag."
```

---

## Red Flags by Factor

| Factor | Red Flag | Yellow Flag |
|--------|----------|------------|
| **days_in_current_stage** | >100 days | >60 days |
| **days_since_latest_meeting** | >30 days | >14 days |
| **meeting_quiet_frac** | >0.8 (80% dark) | >0.5 (50% dark) |
| **number_of_meetings** | <2 total | <4 total |
| **has_champion** | no | unknown |
| **stall_ratio** | >0.5 (half deal stalled) | >0.3 |

If 2+ red flags: deal is risky even if prob looks OK.

---

## Using This to Prioritize

**Monday morning routine:**

1. **Get the CSV** → `results/deal_rankings_with_explanations.csv`
2. **Filter signal=high** → ~15–25% of 1,912 deals (~250–400)
3. **Sort by acv DESC** → biggest revenue first
4. **Read explanation** for top 10, take action
5. **Set calendar** for medium-signal deals (nurture timing)

**Example:**
```bash
# Shell: get top 20 by ACV in high signal
cat results/deal_rankings_with_explanations.csv \
  | grep ",high$" \
  | sort -t, -k8 -nr \
  | head -20
```

---

## What This Is NOT

- ❌ **Not a guarantee** — 82% prob still fails 18% of the time
- ❌ **Not magic** — model trained on closed deals, open deals are different
- ❌ **Not a replacement for your judgment** — if explanation contradicts what you know, trust yourself
- ❌ **Not real-time** — refreshed weekly Monday morning, not live

---

## What This IS

- ✅ A ranking of 1,912 open deals by likelihood
- ✅ Explainable — each deal shows why
- ✅ Actionable — explanations suggest next steps
- ✅ Calibrated — validates weekly on closed deals from 3 months ago

---

## Example Walkthroughs

### Scenario 1: Pipeline Review Call

**Rep says:** "This deal's been stuck in proposal for 100 days. It's a big one."

**Model says:** 42% close prob, signal=medium
```
Top factors:
  ↓ days_in_current_stage: 100 (stuck)
  ↑ acv: $500k (big deal)
  ↓ days_since_latest_meeting: 23 (last touch 3 weeks ago)
Explanation: "Large deal but stalled in proposal. No activity in 3 weeks. 
Likely waiting on budget approval or legal. Recommend stakeholder call."
```

**Your action:**
- Not dead (42% is real), but not hot
- Blocker is probably external (budget/legal/procurement)
- Next step: rep calls champion to ask "What's blocking approval?"
- If "waiting on legal," ask legal timeline. If "waiting on budget," ask who owns budget?
- Set follow-up: 1 week after expected decision point

---

### Scenario 2: New Deal Coming In

**Deal just kicked off yesterday, initial meeting done.**

**Model says:** 8% close prob, signal=none
```
Top factors:
  ↓ days_in_sales_cycle: 1 (brand new)
  ↓ number_of_meetings: 1 (just kickoff)
  ↓ meeting_quiet_frac: 1.0 (no pattern yet)
Explanation: "Too early to assess. Kickoff meeting complete but no follow-up yet. 
Let mature, revisit in 2 weeks."
```

**Your action:**
- This is **normal**. Every deal starts at 8%.
- Don't force close urgency.
- Let rep run their playbook (discovery calls, demos, etc).
- Revisit in 2 weeks. If still no follow-up meetings → escalate.
- If 3 meetings scheduled → come back, prob will jump to 30–40%.

---

### Scenario 3: Deal Looks Good But Something's Off

**Model says:** 71% close prob, signal=high
```
Top factors:
  ↑ meetings_per_month: 3.2
  ↑ has_champion: yes
  ↑ days_in_current_stage: 31 (normal for size)
```

**But you know:** Champion just left the company, new buyer unfamiliar.

**What to do:**
- Model doesn't know the turnover yet (data is 1 week old)
- **Override the model**, flag as risky
- New champion = restart. Update activity, set expectations.
- Check back in 1 week after new champion is onboarded.

**Key:** Model is a starting point, not final word.

---

## Calibration: How Confident Are We?

**Weekly backtest result:**

Model prediction | Actual close rate | Gap
---|---|---
70%+ prob deals | 68% actually close | ✓ Well-calibrated
50–70% prob deals | 52% actually close | ✓ OK
30–50% prob deals | 35% actually close | ✓ OK
10–30% prob deals | 15% actually close | ✓ OK
<10% prob deals | 6% actually close | ✓ OK

Model is slightly underconfident (predicts 68%, sees 70% actual). Safe bias.

---

## Troubleshooting

### Q: "Why is this deal 30% but I'm 90% confident?"

Possible reasons:
1. Deal is newer than model expects (train data is older)
2. Key factor is missing (custom champion relationship, exec sponsor, etc)
3. Model never saw this industry/use case before
4. Your knowledge is more recent (model refreshes weekly, your knowledge is today)

**Action:** Manually override. Send feedback to ML team: "This deal should be 70%+, here's why."

### Q: "Why did a deal's probability drop from 70% to 40% week-to-week?"

Likely reasons:
1. Deal moved into proposal stage (normal dip while blocked on approval)
2. No meetings last week (recency penalty, model thinks it's stalling)
3. Calibrator drifted (monthly refit brought probabilities down)

**Action:** Check `days_since_latest_meeting`. If >14d, push for activity. If just moved stages, give it 1 week.

### Q: "This high-signal deal still didn't close."

1. Life happens (budget pulled, company acquired, CEO change)
2. Our signal was right (70% != 100%), outlier just occurred
3. Rep didn't act on signal (explanation said "close this week," rep did nothing)

**Track it:** Model learns from these. Every closed deal updates future predictions.

---

## Monthly Ops Checklist

- [ ] Monday: Distribute `deal_rankings_with_explanations.csv`
- [ ] Rep actions on top 10 high-signal deals?
- [ ] Any deals dropped from high→low unexpectedly? Investigate.
- [ ] Model recalibrated last week? (check CloudWatch logs)
- [ ] Backtest passed? (ratio <1.30?)
- [ ] Any deal the model predicted wrong? Feedback loop for next month.

---

## Contact

Questions on explanations: See `PROJECT_SUMMARY.md` "Known Limits."  
Want the full technical breakdown? See `BACKTEST_RESULTS.md` and `README.md` "Model Architecture."

Last updated: 2026-08-17

