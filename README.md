# MendSeed AI & QuietFlay v0.1 Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg?logo=pydantic)](https://docs.pydantic.dev/)

**MendSeed AI** is the issue-intelligence and temporal evidence ranking layer for [MendSeed](https://github.com/itsjustayush/mendseed)—a community platform that captures unstructured real-world problem reports, connects citizens with shared grievances, and distills them into developer-actionable briefs.

---

## 🏛️ Two-Layer Decoupled Architecture

To prevent heavy ML libraries (`torch`, `transformers`, `peft`, `bitsandbytes` — over 4 GB) from bloating user environments and the core MendSeed codebase, the repository is split into two clean layers:

```
mendseed-ai/
├── README.md                      # Architecture, QuietFlay v0.1 math, & quickstart
├── .gitignore                     # Blocks credentials, checkpoints, & raw datasets
├── .agents/rules/                 # Antigravity IDE workspace rules
│   └── mendseed-architecture.md
│
├── sdk/                           # 📦 ZERO-ML CLIENT SDK (For MendSeed Core App)
│   ├── pyproject.toml             # Only depends on: pydantic>=2.0, httpx>=0.27
│   └── mendseed_ai/
│       ├── __init__.py            # Clean exports of schemas, engine, & client
│       ├── contracts.py           # The 5 Pydantic validation schemas
│       ├── quietflay.py           # QuietFlay v0.1 pure-math engine (0 dependencies)
│       └── client.py              # Async HTTP bridge with X-MendSeed-Key auth
│
├── server/                        # ⚡ 24/7 CLOUD API GATEWAY (FastAPI / Serverless)
│   ├── requirements.txt           # Minimal CPU dependencies (FastAPI, uvicorn, httpx)
│   ├── vercel.json                # Zero-config 24/7 Vercel serverless deployment
│   ├── Dockerfile                 # Optional container deployment (Render, Fly.io, Cloud Run)
│   ├── .env.example               # Configuration template
│   └── main.py                    # Authenticated REST API & proxy
│
└── training/                      # 🧠 GPU TRAINING WORKSHOP (Run on Colab / Kaggle T4)
    ├── requirements.txt           # GPU dependencies (torch, peft, trl, bitsandbytes)
    ├── generate_training_data.py  # Synthetic dataset generator for 5 schemas
    └── train_qlora.py             # T4-optimized FP16 QLoRA fine-tuning script
```

---

## 🧮 QuietFlay v0.1: Temporal Evidence Ranking Engine

Traditional issue boards rely on **raw net votes** ($U - D$) or naive recency decay, which are easily manipulated by Sybil spikes, copy-paste spam, or legacy monopolies.

**QuietFlay v0.1** introduces a dual-kernel Hawkes excitation process with marked event updates to distinguish **transient viral spikes ($H_0$)** from **persistent unresolved friction ($H_1$)**:

### 1. Dual-Decay Branching Mass
For an issue observing event arrivals at times $t_j$ with quality marks $q_j$:
$$x_{m,k}(t) = \sum_{j: t_j < t} q_j \cdot \exp\left(-\frac{t - t_j}{\tau_k}\right), \quad k \in \{f, s\}$$

- **Fast Kernel ($\tau_f = 0.02\text{ days} \approx 28.8\text{ min}$):** Rapidly decays immediate coordination spikes and duplicate bursts.
- **Slow Kernel ($\tau_s = 3.0\text{ days}$):** Retains long-running community friction over weeks.
- **Composite Score:**
  $$\text{Composite} = w_f \cdot x_f(t) + w_s \cdot x_s(t) \quad (w_f = 0.7, w_s = 0.3)$$

### 2. $O(1)$ Constant-Time Streaming Recurrence
QuietFlay updates in constant arithmetic operations per event without rescanning historical database rows:
$$\Delta I = \sum_{k \in \{f,s\}} (\alpha_{1,k} - \alpha_{0,k}) x_k \tau_k [-\operatorname{expm1}(-\Delta / \tau_k)]$$
$$\text{LLR} \leftarrow \text{LLR} - \Delta I + \log\frac{\lambda_1(t^-)}{\lambda_0(t^-)}$$
$$x_k \leftarrow x_k + q_j$$

### 3. Mathematical Guarantees
- **Zero-Dependency Core:** Implemented using Python's native `math.expm1` and `math.log1p` to eliminate catastrophic floating-point cancellation.
- **Provably Subcritical:** Both kernels satisfy $\nu = \alpha_f \tau_f + \alpha_s \tau_s < 1.0$, guaranteeing bounded branching mass and finite cluster sizes.

---

## 📋 The 5 Pydantic Contracts (`mendseed_ai.contracts`)

| Contract | Purpose | Core Fields |
|---|---|---|
| `IssueAnalysis` | Structured extraction of raw text | `category`, `normalized_title`, `summary`, `affected_area`, `reproducible_details`, `confidence` |
| `IssueComparison` | Pairwise duplicate/relatedness comparison | `relationship`, `evidence`, `matching_score`, `meaningful_differences`, `recommended_action` |
| `CommunitySummary` | Synthesis of recurring grievances | `main_problems`, `recurring_themes`, `disagreements`, `evidence_backed_feedback` |
| `DeveloperBrief` | Technical brief for maintainers | `problem_statement`, `expected_behavior`, `actual_behavior`, `impact`, `reproduction_steps` |
| `MergeProposal` | Guarded deduplication proposal | `candidate_canonical_issue_id`, `proposed_duplicate_links`, `supporting_evidence`, `human_review_required=True` |

---

## 🚀 Quickstart: Installing the SDK in MendSeed

Install the lightweight SDK directly from GitHub into your MendSeed workspace:

```bash
pip install "git+https://github.com/itsjustayush/mendseed-ai.git#subdirectory=sdk"
```

### Usage in MendSeed Outbox Worker:

```python
import os
from mendseed_ai import MendSeedAIClient, QuietFlayEngine, IssueAnalysis

# 1. Initialize Client
ai_client = MendSeedAIClient(
    endpoint_url=os.environ["MENDSEED_AI_URL"],
    api_key=os.environ["MENDSEED_AI_SECRET_KEY"]
)

# 2. Ingest & Analyze Report Asynchronously
analysis: IssueAnalysis = await ai_client.analyze_report(
    "Map locks up when zooming past level 16 on Android 13."
)

# 3. Calculate QuietFlay Temporal Evidence Score (Zero Latency, Pure Python)
quietflay = QuietFlayEngine()
events = [(0.0, 1.0), (0.01, 0.85)]  # (timestamp_in_days, quality_mark)
ranking = quietflay.evaluate_branching_mass(events, current_t=0.05)
print(ranking)
# Output: {'fast_mass': 0.190353, 'slow_mass': 1.772876, 'composite_score': 0.66511}
```

---

## ☁️ Deploying the 24/7 Cloud API Server

### Option A: Vercel Serverless (Recommended — Free & No Cold Spins)
1. Install the Vercel CLI: `npm i -g vercel`
2. Change into the `server/` directory:
   ```bash
   cd server
   vercel
   ```
3. Add your environment variables in the Vercel dashboard:
   - `MENDSEED_AI_SECRET_KEY`: Your private API secret
   - `HF_TOKEN`: Hugging Face Write Token
   - `HF_ENDPOINT_URL`: Your Hugging Face serverless or dedicated model endpoint URL

### Option B: Docker / Render / Cloud Run
```bash
docker build -t mendseed-ai -f server/Dockerfile .
docker run -p 8000:8000 -e MENDSEED_AI_SECRET_KEY=my_key mendseed-ai
```

---

## 🔒 Security & Verification

All API calls require the `X-MendSeed-Key` header, validated with constant-time HMAC comparison:
```bash
curl -X POST https://your-domain.vercel.app/v1/analyze \
  -H "X-MendSeed-Key: your_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "App crashes on launch when Bluetooth permissions are denied."}'
```
