# Cost Comparison: Deployment Options

Compare pricing across deployment methods for the win-probability forecast model.

---

## Monthly Cost Summary

| Option | Compute | Storage | Total | Note |
|---|---|---|---|---|
| **Local (Dev)** | $0 | $0.50 | **$0.50** | Run on laptop, no cloud |
| **Fargate** | $0.02 | $0.55 | **$0.57** | Serverless, pay-per-second |
| **SageMaker** | $2.00 | $1.50 | **$3.50** | Managed ML platform |
| **EC2 (t3.small always-on)** | $12/mo | $0.50 | **$12.50** | Fixed cost, always running |

---

## Detailed Breakdown

### Option 1: Local (Current)

**Setup:** Run scripts locally on laptop, manually or via cron

| Component | Cost | Frequency |
|---|---|---|
| AWS S3 storage | $0.50/mo | Always |
| Compute | $0 | Manual |
| Data transfer | Free (within AWS) | — |
| **Total** | **$0.50/mo** | — |

**Pros:**
- No cloud infrastructure needed
- Full control
- Zero compute cost

**Cons:**
- Laptop must be on
- Manual triggers
- No backup if machine fails
- Hard to scale

**Use case:** Development, ad-hoc testing

---

### Option 2: Fargate (Recommended)

**Setup:** Docker container on ECS Fargate, triggered by EventBridge

| Component | Usage | Cost |
|---|---|---|
| Fargate vCPU-hours | 0.45 vCPU-hours/mo | $0.021 |
| Fargate memory-hours | 2.2 GB-hours/mo | $0.011 |
| S3 storage | 1GB | $0.50 |
| CloudWatch logs | ~5MB/mo | $0.05 |
| Lambda invocations | 5 calls/mo | Free |
| EventBridge rules | 3 rules | Free |
| ECR repo | First repo | Free |
| **Total** | | **$0.57/mo** |

**Breakdown per task:**
```
Weekly scoring (4×):   0.25 vCPU, 512MB, 2 min  → $0.004/run × 4 = $0.016
Monthly calibrate (1×): 0.5 vCPU, 1GB, 5 min   → $0.009/run × 1 = $0.009
Yearly retrain (1×):   1 vCPU, 2GB, 20 min     → $0.047/run ÷ 12 = $0.004
Storage + logs:                                → $0.55
```

**Pros:**
- Cheapest for this workload
- Fully automated (cron)
- No infrastructure to manage
- Scales automatically
- Pay only when running
- CloudWatch logs built-in

**Cons:**
- Slightly more complex setup
- AWS-specific

**Use case:** Production, low-cost, fully automated

---

### Option 3: SageMaker

**Setup:** SageMaker Training Jobs + Model Registry

| Component | Usage | Cost |
|---|---|---|
| Weekly batch transform | 4 jobs × 5 min | $0.04 |
| Monthly training job | 1 job × 5 min | $0.01 |
| Yearly retrain | 1 job × 20 min | $0.001 |
| S3 storage | 1GB | $1.00 |
| CloudWatch | Monitoring | $0.50 |
| Model registry | Storage | Free |
| **Total** | | **$1.56/mo** |

**Pros:**
- Native ML platform (built for this)
- Automatic versioning
- Built-in monitoring
- Experiment tracking
- Model registry

**Cons:**
- 2.7× more expensive than Fargate
- Over-engineered for our use case
- Slower setup

**Use case:** Large ML teams, heavy experimentation

---

### Option 4: EC2 (t3.small Always-On)

**Setup:** Always-running EC2 instance, manual triggers via SSH

| Component | Usage | Cost |
|---|---|---|
| EC2 t3.small | 730 hours/mo | $12.00 |
| EBS storage | 20GB | $2.00 |
| Data transfer | Outbound | $0.50 |
| Elastic IP | 1 | $3.65 |
| **Total** | | **$18.15/mo** |

**Pros:**
- Full control
- Can SSH in anytime
- Simple setup

**Cons:**
- 32× more expensive than Fargate
- Fixed cost (runs even if idle)
- Maintenance burden
- Must patch OS

**Use case:** Not recommended for this workload

---

### Option 5: Lambda (Limited)

**Setup:** Lambda functions triggered by EventBridge

| Component | Usage | Cost |
|---|---|---|
| Lambda invocations | 5 calls/mo | Free (1M/mo included) |
| Lambda compute | Limited to 15 min | Can't run full retrain |
| S3 storage | 1GB | $0.50 |
| **Total** | | **$0.50/mo** |

**Pros:**
- Cheapest compute option
- Zero infrastructure
- Fully managed

**Cons:**
- 15 min timeout (can't run yearly retrain)
- 10GB memory limit
- Cold start latency
- Limited for complex workloads

**Use case:** Scoring only (not training)

---

## Year-over-Year Cost

| Option | Year 1 | Year 2+ | Notes |
|---|---|---|---|
| Local | $6 | $6 | Laptop electricity |
| **Fargate** | **$7** | **$7** | No setup fees |
| SageMaker | $18.72 | $18.72 | After $2k free tier |
| EC2 | $217 | $217 | Includes IP address |
| Lambda (scoring only) | $6 | $6 | Missing retrain |

---

## Recommendation by Use Case

### Development / Ad-hoc
→ **Local ($0.50/mo)**
- Run manually from laptop
- No cloud setup needed
- Good for testing

### Production / Small Scale
→ **Fargate ($0.57/mo)** ✅ RECOMMENDED
- Fully automated
- Cheapest serverless option
- Reliable, no maintenance
- Scales if needed

### Production / Large Scale (100+ jobs/day)
→ **SageMaker ($1.56/mo)**
- Experiment tracking
- Model versioning
- Team collaboration

### Production / Full Inference API
→ **SageMaker Endpoint ($84+/mo)**
- Always-on API
- <100ms latency
- Real-time scoring

---

## Implementation Effort

| Option | Setup Time | Complexity | Ongoing |
|---|---|---|---|
| Local | 10 min | Trivial | 5 min/week |
| **Fargate** | **1–2 hours** | **Medium** | Automated |
| SageMaker | 2–3 hours | High | Automated |
| EC2 | 1 hour | Low | 30 min/month patching |

---

## Migration Path

**Start with Local, migrate to Fargate when:**
- Need scheduled (not manual) execution
- Want to eliminate dependency on laptop
- Ready for production operations

**From Fargate to SageMaker when:**
- Building team of ML engineers
- Heavy experimentation / A-B testing
- Need production inference endpoint

---

## Final Recommendation

**Use Fargate for this project.**

**Why:**
- ✅ Cheapest for current workload ($0.57/mo)
- ✅ Fully automated, no manual triggers
- ✅ No infrastructure maintenance
- ✅ Scales automatically
- ✅ Can add real-time API later ($84/mo for endpoint)
- ✅ Easy to migrate to SageMaker later

**Setup:**
1. Follow FARGATE_DEPLOYMENT.md (1–2 hours one-time)
2. Zero monthly cost increase beyond S3 storage
3. Fully automated forever

**Risk:** Essentially none. Can revert to local in 30 minutes.

