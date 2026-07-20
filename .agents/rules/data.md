---
trigger: always_on
---

# AGENTS.md — Forensic Fraud Detection System

This file defines the architecture, scope, and guardrails for building this project with an AI coding agent. Follow the phases in order. Do not skip ahead. Do not add features not listed in a phase without updating this file first.

## Project Summary

A forensic fraud detection system that combines:
- **Isolation Forest** (unsupervised anomaly detection) — flags statistical outliers without needing labeled fraud
- **XGBoost** (supervised classification) — learns from labeled fraud/non-fraud history
- **Benford's Law** — flags transactions/amounts whose leading-digit distribution deviates from expected, a classic forensic accounting signal
- **Compliance engine** — maps flagged transactions to relevant UK regulatory triggers (e.g. Money Laundering Regulations 2017, FCA SYSC requirements) so output isn't just "this is anomalous" but "this is anomalous AND here's why it matters regulatorily"

The end deliverable is not a notebook. It's a runnable pipeline + a simple dashboard/report a non-technical reviewer (interviewer) can look at and understand in under 2 minutes.

## Guardrails (read before every phase)

1. **No phase starts until the previous phase's Definition of Done is met.** If the agent wants to jump ahead (e.g. build the dashboard before the model works), it must say so explicitly and get confirmation first.
2. **Every modeling decision must be explainable in one sentence.** If the agent can't justify a choice (why Isolation Forest AND XGBoost, why this contamination rate, why this threshold) in plain English, it should not make that choice silently — it should surface the tradeoff.
3. **Do not fabricate data provenance.** If using a synthetic or public dataset (e.g. Kaggle credit card fraud dataset), say so clearly in the README. Never imply real transaction data was used.
4. **Do not over-claim compliance accuracy.** The compliance engine maps flags to regulatory *categories* for illustrative/portfolio purposes — the README must state this is not legal/compliance advice.
5. **Keep the interview-defensibility bar in mind.** After each phase, the agent should be able to answer: "What would you say if an interviewer asked how this works?" If the answer is "I'm not sure," stop and simplify.
6. **No silent dependency bloat.** New libraries require a one-line justification in the commit message or PR description.

---

## Phase 0 — Setup & Scope Lock

**Goal:** Environment ready, dataset chosen, scope frozen.

- [ ] Initialize repo structure:
  ```
  /data          (raw + processed, gitignored if large)
  /notebooks     (exploration only, not final logic)
  /src
    /features
    /models
    /compliance
    /utils
  /reports       (generated outputs, figures)
  /app           (dashboard, if applicable)
  README.md
  AGENTS.md
  requirements.txt
  ```
- [ ] Choose dataset (recommend: Kaggle "Credit Card Fraud Detection" — anonymized real transactions, or synthetic equivalent). Document choice and its limitations in README.
- [ ] Set up virtual environment and `requirements.txt` (pandas, numpy, scikit-learn, xgboost, matplotlib/seaborn, streamlit or dash for the app).
- [ ] Write one paragraph in README: the business problem this solves and who the fictional "user" of this system is (e.g. a mid-size bank's compliance analyst).

**Definition of Done:** Repo scaffolded, dataset downloaded and documented, README has a problem statement. Nothing modeled yet.

---

## Phase 1 — Data Understanding & Cleaning

**Goal:** Know the data cold before touching a model.

- [ ] Load data, document shape, class balance (fraud is almost always <1% — state the exact ratio).
- [ ] Check and handle: missing values, duplicate transactions, data type issues.
- [ ] Basic EDA: distribution of transaction amounts, time-based patterns, class imbalance visualization.
- [ ] Save a cleaned dataset to `/data/processed/`.

**Definition of Done:** A short EDA report (markdown or notebook) exists summarizing what the data looks like and what problems it has (imbalance, skew, etc). Agent can state the fraud rate from memory.

---

## Phase 2 — Benford's Law Module

**Goal:** A standalone, testable module that flags leading-digit anomalies.

- [ ] Implement leading-digit extraction on transaction amounts.
- [ ] Compare observed distribution to Benford's expected distribution (chi-square or MAD test).
- [ ] Output a per-transaction or per-group "Benford deviation score."
- [ ] Unit test this module in isolation with a synthetic example known to violate Benford's Law.

**Definition of Done:** `src/features/benford.py` runs standalone, has a test, and produces a score column. Agent can explain in one sentence why leading-digit distributions matter for fraud (people faking numbers tend to produce unnatural digit patterns).

---

## Phase 3 — Feature Engineering

**Goal:** Build the feature set both models will use.

- [ ] Transaction-level features: amount, time-of-day/week patterns, frequency features (e.g. transactions per account per hour).
- [ ] Merge in the Benford deviation score from Phase 2.
- [ ] Handle class imbalance strategy decision (SMOTE, class weights, or undersampling) — document the choice and why.
- [ ] Train/test split with stratification, time-aware split if the data has timestamps (avoid leakage).

**Definition of Done:** A single clean feature matrix ready for both unsupervised and supervised modeling. README/notes state the imbalance strategy and why it was chosen over alternatives.

---

## Phase 4 — Unsupervised Model: Isolation Forest

**Goal:** Anomaly scores independent of labels.

- [ ] Train Isolation Forest on features (excluding label).
- [ ] Tune contamination parameter with justification (don't just default it — explain the choice against the known fraud rate).
- [ ] Generate anomaly scores for all transactions.
- [ ] Evaluate against known labels *only as a sanity check* (this model should also work if labels didn't exist).

**Definition of Done:** Isolation Forest scores saved, with a short writeup of what "contamination" means and how it was set.

---

## Phase 5 — Supervised Model: XGBoost

**Goal:** A tuned classifier with honest evaluation.

- [ ] Train XGBoost on labeled data, using the imbalance strategy from Phase 3.
- [ ] Evaluate with precision, recall, F1, and PR-AUC (not just accuracy — accuracy is meaningless on <1% fraud rate).
- [ ] Plot confusion matrix and discuss false positive vs false negative cost tradeoff in plain language.
- [ ] Feature importance analysis — which features actually drive fraud predictions.

**Definition of Done:** Model saved, metrics documented, and the agent (and user) can explain the precision/recall tradeoff choice in interview-ready language — specifically why recall might matter more than precision in fraud (or vice versa, and why).

---

## Phase 6 — Ensemble / Combined Signal

**Goal:** Combine Isolation Forest + XGBoost + Benford score into one final risk score.

- [ ] Define a combination logic (e.g. weighted score, or flag-if-any-two-trigger rule). Document the reasoning — don't just average blindly.
- [ ] Produce a final `risk_score` and `risk_tier` (e.g. Low/Medium/High) per transaction.

**Definition of Done:** One function takes raw transaction features and returns a final risk tier. This is the core "product" of the project.

---

## Phase 7 — Compliance Engine

**Goal:** Map high-risk flags to relevant UK regulatory categories.

- [ ] Build a simple rule-mapping layer: e.g. high-risk + structuring pattern → Money Laundering Regulations 2017 trigger; high-risk + repeated small transactions → SAR (Suspicious Activity Report) consideration.
- [ ] Keep this rule-based and transparent, not a black box — it should be a lookup/mapping table, documented and easy to audit.
- [ ] Add the disclaimer (README + in-app if applicable): this is illustrative, not legal/compliance advice.

**Definition of Done:** Given a flagged transaction, the system outputs which regulatory category it maps to and a one-line reason.

---

## Phase 8 — Reporting / Dashboard

**Goal:** A reviewer can see results without reading code.

- [ ] Build a lightweight Streamlit (or similar) app: upload/select transactions → see risk scores, risk tiers, compliance flags, and key charts (fraud distribution, feature importance, Benford deviation chart).
- [ ] Keep UI simple — this is a portfolio demo, not a production compliance tool.

**Definition of Done:** App runs locally with one command, shows end-to-end results on sample data.

---

## Phase 9 — Documentation & Interview-Readiness

**Goal:** Make this defensible.

- [ ] Final README: problem, data source + limitations, architecture diagram (even simple ASCII/markdown), how to run it, results summary, and a "Limitations & Next Steps" section (be honest — e.g. "would use real transaction data and expert-labeled compliance mappings in production").
- [ ] Write a short "How I'd explain this in an interview" section — 5-6 sentences, in plain English, no jargon.
- [ ] Push to GitHub with a clean commit history (not one giant commit).

**Definition of Done:** A stranger could clone the repo, read the README, and understand what was built and why in under 3 minutes.

---

## Phase 10 — Polish (optional, only if time allows)

- [ ] Add basic tests for the pipeline (not just Benford module).
- [ ] Add a requirements-pinned `environment.yml` or Docker setup for reproducibility.
- [ ] Deploy the dashboard (Streamlit Community Cloud or similar) so it's a live link, not just code.

---

## Agent Operating Notes

- Work phase by phase. At the end of each phase, summarize in 2-3 sentences what was built and ask before moving to the next phase.
- If a phase's scope creeps (e.g. "let's also add real-time streaming"), flag it as out of scope and note it under a "Future Ideas" section in README instead of building it.
- Prioritize correctness and explainability over cleverness. A simpler model the user can explain beats a fancier one they can't.