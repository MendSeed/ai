# MendSeed AI & QuietFray v0.1 Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg?logo=pydantic)](https://docs.pydantic.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-30%2F30%20Passed-brightgreen.svg)]()

**MendSeed AI** powers [MendSeed](https://github.com/MendSeed)—a **public grievance observatory and real-world problem discovery platform**.

MendSeed listens to the lived grievances and everyday struggles of citizens across different sectors (agriculture, civic infrastructure, healthcare, education, sanitation, transport, local governance, livelihoods) and structures them into **actionable problem briefs for builders, student researchers, and hackathon participants** seeking genuine problems worth solving.

---

## 🏛️ Architecture & Service Topology

The service is decoupled into two distinct operational layers:

```
┌──────────────────────────────────────┐
│  MendSeed Core Platform / Frontend  │
└──────────────────┬───────────────────┘
                   │
                   ▼  (X-MendSeed-Key Authentication)
┌────────────────────────────────────────────────────────┐
│  MendSeed AI Gateway (FastAPI / Serverless / CPU)      │
│  - Port 8000 (Vercel / Cloud Run / Docker)            │
│  - Input boundary validation & rate limiting           │
│  - QuietFray v0.1 Pure-Math Ranking Engine (O(1))      │
│  - Provider abstraction & fallback management          │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼  (Bearer Token / Dedicated VPC)
┌────────────────────────────────────────────────────────┐
│  Remote GPU Inference Worker (Hugging Face / vLLM)    │
│  - Hosted 3B Parameter Model + QLoRA Adapter           │
│  - Independent warm/cold lifecycle                     │
└────────────────────────────────────────────────────────┘
```

### Critical Reality: Gateway vs. Model Inference Worker
1. **The Gateway is Lightweight CPU:** The FastAPI server (deployed to Vercel, Docker, or Cloud Run) **never** loads or hosts the 3-billion-parameter model weights locally. It acts as an always-reachable, authenticated orchestrator.
2. **Cold Starts vs. Warm Endpoints:** Serverless or free GPU endpoints go cold after periods of inactivity. The gateway explicitly handles `503 Model Loading` states (returning estimated wait times) and falls back safely to domain heuristics, tagging responses with `X-MendSeed-Inference-Source: heuristic-fallback`. We do **not** claim free 24/7 warm GPU inference unless dedicated hardware is provisioned and funded.

---

## 🔐 Security & API Correctness (Phase A Foundations)

1. **Mandatory Explicit Authentication:**
   - Protected endpoints require an explicit `MENDSEED_AI_SECRET_KEY`.
   - If the environment variable is missing or empty, the server **fails safely** with `HTTP 500 Server Configuration Error` instead of falling back to an insecure default key.
   - Header verification uses constant-time `secrets.compare_digest` to prevent timing attacks.
2. **Private Public Health Checks:**
   - `/health` is a public liveness probe reporting service readiness.
   - It **never** leaks internal secrets, environment variable presence (`env_secret_active`), or upstream tokens.
3. **CORS Allowlist:**
   - Wildcard credentialed CORS (`allow_origins=["*"]` with `allow_credentials=True`) has been completely removed.
   - Origins are strictly configured via `MENDSEED_CORS_ORIGINS`. Browser JavaScript must never call this private gateway directly using the service key.
4. **Clean Secret Hygiene:**
   - `.env.example` contains only safe placeholders (`your_secret_key_here`, `hf_token_placeholder`).
5. **Pydantic Validation & Fallback Distinguishability:**
   - When upstream model inference succeeds:
     - `X-MendSeed-Inference-Source: model-inference`
     - `X-MendSeed-Fallback: false`
   - When a fallback occurs (due to timeout, model loading, or malformed JSON):
     - `X-MendSeed-Inference-Source: heuristic-fallback`
     - `X-MendSeed-Fallback: true`
     - `X-MendSeed-Fallback-Reason: <status>`
   - A heuristic output is **never** misrepresented as a neural model result.

---

## 🧮 QuietFray v0.1: Deterministic Evidence Ranking Engine

Traditional community platforms rely on **raw net votes** ($U - D$) or naive recency decay, which are easily gamed by bot swarms and viral outrage while ignoring chronic, quiet suffering.

**QuietFray v0.1** uses a bivariate dual-kernel Hawkes point process with quality-weighted marks ($q_j \in [0, 1]$) to separate **transient viral spikes ($H_0$)** from **persistent unresolved community friction ($H_1$)**:

### Mathematical Specification
1. **Dual-Decay Branching Mass:**
   $$x_k(t) = \sum_{j: t_j < t} q_j \cdot \exp\left(-\frac{t - t_j}{\tau_k}\right), \quad k \in \{f, s\}$$
   - **Fast Kernel ($\tau_f = 0.02\text{ days} \approx 28.8\text{ mins}$):** Dissipates coordinated bot bursts and duplicate spikes.
   - **Slow Kernel ($\tau_s = 3.0\text{ days}$):** Retains multi-week community friction.
   - **Composite Score:** $\text{Score} = 0.7 \cdot x_f(t) + 0.3 \cdot x_s(t)$.
2. **Provable Subcriticality:**
   Branching ratio $\nu = \alpha_f \tau_f + \alpha_s \tau_s < 1.0$ is strictly verified at initialization across all channels ($\nu_{r0}=0.360, \nu_{r1}=0.550, \nu_{v0}=0.315, \nu_{v1}=0.487$). The Hawkes process is provably non-explosive.
3. **$O(1)$ Streaming State Recurrence:**
   Updates in constant arithmetic operations per event using native `math.expm1` and `math.log1p`, avoiding floating-point cancellation.

### Measured Ranking Benchmark (Real NDCG@10)
Rather than using fabricated numbers, the benchmark ranks a candidate pool of real community problems, legacy topics, bot bursts, and chatter against explicit ground-truth relevance grades:

| Ranking Algorithm | Measured NDCG@10 | Sybil Burst Rank (100 spam votes) | Chronic Need Rank (5 reports over 14d) |
|---|:---:|:---:|:---:|
| **QuietFray v0.1** | **0.7501** | Suppressed to lower ranks | **Rank #1 (Elevated)** |
| **Raw Net Votes** | 0.6209 | Rank #1 (Distorted feed) | Ignored / Buried |
| **Wilson Score** | 0.6481 | Rank #1 (Distorted feed) | Low rank |
| **Recency Decay** | 0.6812 | Transient top rank | Medium rank |

---

## 📋 The 5 Pydantic Contracts (`mendseed_ai.contracts`)

| Contract | Purpose | Core Fields |
|---|---|---|
| `GrievanceAnalysis` | Normalizes raw citizen reports across sectors | `category`, `normalized_title`, `summary`, `affected_sector`, `observed_friction`, `confidence` |
| `GrievanceComparison` | Evaluates shared root causes vs distinct issues | `relationship`, `evidence`, `matching_score`, `meaningful_differences`, `recommended_action` |
| `CommunitySummary` | Synthesizes recurring themes across submissions | `main_problems`, `recurring_themes`, `disagreements`, `evidence_backed_feedback` |
| `BuilderBrief` | **Actionable problem brief for hackathons & projects** | `problem_statement`, `target_beneficiaries`, `current_ground_reality`, `solution_opportunities`, `unresolved_questions` |
| `CanonicalClusterProposal` | Cautious problem clustering proposal | `candidate_canonical_problem_id`, `proposed_grievance_links`, `supporting_evidence`, `human_review_required=True` |

---

## 🔬 Dataset & Training Workflow (Phase C)

### Provenance & Honesty Policy
- **Synthetic Seed Baseline:** Current repository includes seed examples (`training/mendseed_training_data.jsonl` and `training/data/`) labeled with `origin: "synthetic_seed"` and `quality_verified: False`.
- **Pilot Data Plan (`training/pilot_data_plan.md`):** Targets **475 curated records** (300 Train / 75 Validation / 100 Locked Test) across all 5 tasks and 8 sectors.
- **Dataset Validator (`training/validate_dataset.py`):**
  - Enforces schema and contract conformity for every record.
  - Detects split leakage via cluster ID sharing and word Jaccard similarity ($J > 0.65$).
  - Audits sector coverage across all 8 mandatory public domains.

### QLoRA Training Pipeline (`training/train_qlora.py`)
- **CUDA Hardware Preflight:** Refuses to run on CPU machines with clear guidance to launch on hosted GPU hardware.
- **Real Validation Checkpointing:** Evaluates on validation split (`eval_strategy="steps"`) and selects best checkpoint via validation loss, removing hardcoded step 75.
- **Audit Logging:** Saves `training_metadata.json` with base model commit hash, training data SHA-256 checksum, seed, and trainer history.
- **Honest Status:** If no GPU trainer history is present, loss curves are reported as **`NOT RUN`** rather than simulated.

---

## 🚀 Local Verification & Test Commands

### 1. Run Complete Unit Test Suite (30 Tests)
```bash
py -3 -m pytest tests/ -v
```
Verifies security, authentication, health check privacy, provider mocks, input bounds, fallback headers, QuietFray math, and SDK client.

### 2. Run Comprehensive Benchmark Suite
```bash
py -3 benchmark/run_benchmarks.py --provider mock
```
Evaluates schema pass rate, sector accuracy, adversarial defense, and measured QuietFray NDCG@10.

### 3. Validate Training Data & Check for Leakage
```bash
py -3 training/validate_dataset.py training/mendseed_training_data.jsonl
```

### 4. Hosted GPU Training Run (Google Colab / Kaggle / Cloud GPU)
To execute one reproducible fine-tuning run on a GPU node:
```bash
# In hosted GPU session with NVIDIA GPU:
pip install -r training/requirements.txt
pip install -e sdk/

export BASE_MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
export HF_TOKEN="hf_your_token_here"

# Execute training with validation checkpointing
python training/train_qlora.py \
  --train-file training/data/train.jsonl \
  --val-file training/data/val.jsonl \
  --max-steps 150 \
  --eval-steps 15 \
  --push-to-hub
```
