# SHAP + LLM Explanations TODO

Status: SHAP extraction done. LLM prompts pending.

---

## Completed ✓

- [x] `explain_predictions.py` — Extract SHAP feature importance
  - TreeExplainer (native CatBoost, ~2 min for 1,912 deals)
  - Output: `models/shap_explanations.csv` with top 3 factors per deal
  - Ready to test: `python explain_predictions.py`

---

## Pending

### Step 1: Test SHAP Extraction (5 min)

```bash
python explain_predictions.py
```

Verify output:
- `models/shap_explanations.csv` created
- ~1,912 rows
- Columns: display_id, win_probability, top_factor_1/2/3, top_factor_*_impact

### Step 2: Add Claude API Integration (30 min)

Create `generate_llm_explanations.py`:

```python
from anthropic import Anthropic
import pandas as pd

def generate_explanation(deal):
    """1-2 sentence advice per deal using Claude."""
    
    client = Anthropic()
    
    top_factors_text = f"""
Top factors:
- {deal['top_factor_1']} (SHAP: {deal['top_factor_1_impact']:.2f})
- {deal['top_factor_2']} (SHAP: {deal['top_factor_2_impact']:.2f})
- {deal['top_factor_3']} (SHAP: {deal['top_factor_3_impact']:.2f})
"""
    
    prompt = f"""Sales rep needs 1-2 sentence actionable advice on this deal. Be direct.

Deal ID: {deal['display_id']}
Win probability: {deal['win_probability']:.1%}
Signal strength: {deal['signal_strength']}

{top_factors_text}

Advice (1-2 sentences, no filler):"""
    
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()

# Load SHAP results
shap_df = pd.read_csv("models/shap_explanations.csv")

# Generate explanations (batch, ~1-2 sec per deal = 30-60 min for 1,912)
# Recommend: start with top 50, measure cost, then decide full batch
explanations = []
for idx, row in shap_df.head(50).iterrows():
    expl = generate_explanation(row)
    explanations.append({
        'display_id': row['display_id'],
        'llm_explanation': expl
    })

# Save
pd.DataFrame(explanations).to_csv("results/deal_explanations_sample.csv", index=False)
```

**Cost estimate:** ~$2.60 per full run (1,912 deals)

### Step 3: Merge SHAP + LLM into Single CSV (10 min)

```python
shap_df = pd.read_csv("models/shap_explanations.csv")
llm_df = pd.read_csv("results/deal_explanations_sample.csv")

merged = shap_df.merge(llm_df, on="display_id", how="left")
merged.to_csv("results/deal_rankings_with_explanations.csv", index=False)
```

Output columns:
- display_id
- win_probability
- signal_strength
- acv
- top_factor_1/2/3 (SHAP features)
- llm_explanation (actionable text)

### Step 4: Add to Fargate Pipeline (20 min)

Update `Dockerfile`:
```dockerfile
RUN pip install shap anthropic
```

Update Lambda trigger (new `forecast-explain` task):
```python
# Run AFTER weekly scoring
ecs.run_task(
    cluster='forecast-cluster',
    taskDefinition='forecast-explain:1',  # NEW
    launchType='FARGATE',
    ...
)
```

Add ECS task definition:
```json
{
  "family": "forecast-explain",
  "cpu": "512",
  "memory": "1024",
  "command": ["explain_predictions.py"],  # SHAP only (fast)
}
```

**Note:** Run SHAP weekly, LLM explanations monthly-on-demand (save $30/mo)

### Step 5: Monitoring & Alerts (optional)

Add to `track_predictions.py`:
```python
# After each explain run, log to CloudWatch
print("EXPLANATIONS_GENERATED: 1912 deals")
print(f"LLM_COST: ${deal_count * 0.0014:.2f}")
```

---

## Rollout Timeline

| Phase | Time | What | Cost |
|-------|------|------|------|
| 1 | Today | Run `explain_predictions.py` test | $0 |
| 2 | This week | Generate LLM on top 50 deals (sample) | $0.13 |
| 3 | Next week | Full LLM batch if needed | $2.60 |
| 4 | After validation | Add to weekly Fargate cron | +$0.01/week SHAP |
| 5 | (Optional) | Monthly LLM refresh instead of weekly | +$2.60/month |

---

## Risk / Backout

- SHAP is local compute, can't fail (only gets slow)
- LLM calls are stateless, retry-safe
- If Claude API cost explodes: switch to batch-per-month (not weekly)
- If explanations bad: revert to SHAP factors only (CSV keeps working)

---

## Questions Before Starting

1. **Scope:** Full 1,912 deals, or top 50 sample first?
2. **Frequency:** Weekly (automated, $10/mo) or monthly on-demand ($2.60/mo)?
3. **Output:** Single CSV with SHAP + LLM, or separate CSVs?

