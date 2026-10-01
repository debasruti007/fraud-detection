# Claim Fraud-Risk Scoring & Fairness Audit

A small web app that scores insurance claims for fraud risk, explains each
score with SHAP, and audits the model's decisions for demographic bias.

**Live demo:** https://fraud-detection-frontend-nu.vercel.app
**API:** https://fraud-detection-sv8u.onrender.com/docs

> Render's free tier sleeps after inactivity — the first request after a
> while may take 30-60 seconds to respond.

---

## Why this project

Bajaj Allianz Life won **"Best Fraud Prevention Platform"** at the FinTech
India Innovation Awards 2022, for a fraud model that profiles customers
using **customer and geographical demographics combined with historical
fraudulent claims and case data**.
([source](https://www.nkgsb-bank.com/pdf/BALIC.pdf))

Separately, Bajaj Allianz Life's Chief Information and Digital Officer,
Goutam Datta, described using a **"shadowing technique"** — human oversight
layered on top of AI-driven decisions — plus data vetting, specifically to
reduce AI bias in their models.
([source](https://www.cio.inc/how-bajaj-allianz-life-insurance-mitigating-ai-biases-a-23934))

This project is not a reconstruction of Bajaj's actual system — their real
feature set, data, and architecture are not public. It is a project built
on the same two stated principles, with every design choice backed by
evidence rather than assumption:

- **The fraud model** profiles claims using geographic fields
  (`policy_state`, `incident_state`, `incident_city`) combined with claim
  and incident history (`incident_severity`, `total_claim_amount`, etc.) —
  mirroring the award's "demographics + geography + case history" framing.
- **The fairness audit**, and the decision to route flagged claims to a
  human reviewer rather than auto-rejecting them, mirrors the "shadowing"
  concept and the stated concern about bias.

## What it does

1. **Fraud scoring** — enter claim details, get a fraud probability, a
   flagged/not-flagged decision, and a SHAP chart showing which fields
   drove the score.
2. **Fairness dashboard** — breaks down flag rates, real fraud rates, and
   false-flag rates by policy state, sex, age bracket, and incident state,
   with a disparate-impact ("80% rule") check per group.

## Data

Public synthetic dataset, ~1,000 auto insurance claims
([source](https://github.com/mwitiderrick/insurancedata)). No real,
labeled insurance-fraud dataset is publicly available anywhere — real
claims data involves identifiable people and active investigations, so
every public option in this space, including this one, is synthetic. This
is disclosed rather than hidden.

## Modeling decisions, and the evidence behind each one

Every choice below was tested with 5-fold cross-validation on an 800-row
training split, with a 200-row test set held out and scored only once per
feature-set version.

| Decision | Evidence |
|---|---|
| Dropped `insured_zip`, `incident_location` | 995/1000 and 1000/1000 unique values — too unique to generalize from |
| Dropped `auto_model` | Tested in/out; no score improvement |
| Dropped `policy_bind_date`, `incident_date` | Mean gap between the two dates: 4,738 days (non-fraud) vs 4,743 days (fraud) — no signal |
| Kept `total_claim_amount` only, dropped the 3 claim-part columns | Parts sum to the total in 100% of rows — exact duplicates |
| Dropped `age`, `insured_sex` | Removing them cost nothing (0.864 vs 0.859 AUC) — kept out of the model, used only in the fairness audit |
| Dropped `months_as_customer` | Correlates 0.92 with age (a potential proxy); removing it was free |
| Dropped `insured_occupation` | Removing it cost nothing (0.872 vs 0.872 AUC) |
| No class weighting | Weighted version scored slightly worse (0.857 vs 0.859 AUC) |
| XGBoost: depth 3, learning rate 0.03, 100 trees | Best cross-validated average precision among 27 combinations tried; simpler settings tied with more complex ones |
| Threshold = 0.40 | Precision stayed ~0.65 across all thresholds tested; lower thresholds trade little precision for much better recall |

**Final test-set result (scored once, not tuned on):** ROC-AUC 0.824,
average precision 0.587. At threshold 0.40: 33% of claims flagged, 63%
precision, 84% recall.

**Sanity check against a one-line rule** ("flag every Major Damage
claim"): 60% precision, 66% recall. The model beats this, but a large
share of its power comes from one feature (`incident_severity`), which
is disclosed below, not hidden.

## Known limitations (stated up front, not discovered by an interviewer)

- **Synthetic data.** Real insurance fraud data is not publicly available
  for any insurer, for privacy and regulatory reasons.
- **`insured_hobbies` (chess, cross-fit) are the 2nd and 3rd most
  important SHAP features**, likely an artifact of how this synthetic
  dataset was generated — a hobby has no real causal link to fraud. This
  is disclosed as a limitation, not presented as a real signal.
- **Fairness finding: claimants aged 50+ have a false-flag rate of 22.5%**,
  versus ~14% for other age brackets (not-flagged ratio 0.787, below the
  0.80 rule-of-thumb threshold). Age is not a model input. Traced to: honest
  50+ claimants have a higher share of Major Damage incidents (20.0% vs
  13.9%) — the model's strongest feature — rather than to a proxy for age.
  With ~89 honest claims in that group, this result is a genuine finding
  worth reporting, not a confirmed, statistically certain effect.
- **A 33%-of-claims review queue is large** for a real fraud operation.
  This dataset's signal isn't strong enough to shrink it further without
  losing recall.

## Architecture

```
backend/   FastAPI + XGBoost + SHAP, deployed on Render
frontend/  Plain HTML/JS + Chart.js, deployed on Vercel
```

- `/predict` — score a claim, return probability + SHAP explanation
- `/fairness` — group-level fairness report
- `/options`, `/metrics`, `/health` — supporting endpoints

Full feature-selection and tuning process is in `backend/notebooks/01_eda.ipynb`.