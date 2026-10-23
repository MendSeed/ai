---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
- mendseed
- public-grievances
- qlora
- peft
- hackathons
- quietfray
- pydantic
pipeline_tag: text-generation
---

# MendSeed AI — Qwen3-4B Grievance Intelligence Adapter Specification

This model card documents the target architecture, training configuration, evaluation methodology, and operational constraints for fine-tuning **`Qwen/Qwen3-4B-Instruct-2507`** for **MendSeed**—a public grievance observatory and real-world problem discovery platform.

MendSeed structures ground-level citizen grievances across key sectors (agriculture, civic infrastructure, healthcare, education, logistics, sanitation, public services, daily livelihoods) into actionable problem briefs for builders, researchers, and hackathon teams.

---

## Model Details

- **Base Model:** `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0)
- **Adapter Type:** PEFT QLoRA (Rank $r=16$, Alpha $\alpha=32$, Dropout $0.05$)
- **Target Modules:** All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Quantization:** 4-bit NormalFloat (NF4) with Double Quantization (`float16` compute dType)
- **Target Contracts:**
  1. `GrievanceAnalysis` (Structured parameter extraction from messy citizen voice)
  2. `GrievanceComparison` (Pairwise shared root cause analysis)
  3. `CommunitySummary` (Synthesis across multiple community submissions)
  4. `BuilderBrief` (Actionable problem briefs for hackathons and project builders)
  5. `CanonicalClusterProposal` (Cautious clustering with mandatory human review: `human_review_required=True`)

---

## Dataset & Evaluation Status

### Data Provenance Status
- **Current Repository State:** 10 synthetic seed examples partitioned into non-overlapping splits (`train.jsonl`, `val.jsonl`, `test.jsonl`) for schema validation and CI pipeline testing.
- **Pilot Dataset Plan (`training/pilot_data_plan.md`):** Formal plan targeting **475 curated records** (300 Train / 75 Validation / 100 Locked Test) across all 8 sectors. Machine-generated synthetic seeds are explicitly labeled and **never** represented as gold-standard or human-verified.

### Evaluation Metrics & Integrity
In accordance with our strict empirical honesty policy, metrics reflect real evaluations rather than fabricated curves:

| Metric Dimension | Result | Methodology / Status |
|---|:---:|---|
| **Pydantic Schema Integrity** | **100.0%** | Validated via `tests/test_endpoints.py` and `tests/test_dataset_validator.py` |
| **QuietFray v0.1 Measured NDCG@10** | **0.7501** | Measured on candidate problem pool (vs **0.6209** for raw net votes) |
| **Sybil Burst Suppression** | **Demonstrated** | 100 burst upvotes in 15 mins suppressed to bottom feed ranks |
| **Optimal Checkpoint & Loss Curves** | **NOT RUN** | Pending hosted GPU training session. Simulated curves are disallowed. |
| **Inference Source Tagging** | **Enforced** | Gateway emits `X-MendSeed-Inference-Source: model-inference` vs `heuristic-fallback` |

---

## Architectural Constraints & Deployment Realities

1. **Gateway Decoupling:** The FastAPI server (`server/main.py`) runs in lightweight CPU/serverless environments (Vercel, Docker, Cloud Run). It **does not** load or execute 3B parameter weights locally.
2. **Cold-Start Handling:** Free/serverless inference endpoints sleep when inactive. The gateway handles upstream 503 model-loading states cleanly and provides estimated wait times rather than advertising free instant 24/7 inference.
3. **Security:** The service key (`MENDSEED_AI_SECRET_KEY`) is stored exclusively in server-side deployment secrets and verified with constant-time HMAC digest comparison. Browser JavaScript must never call this private gateway directly.

---

## Hosted GPU Training Instructions

To execute a reproducible QLoRA fine-tuning session on an NVIDIA GPU (e.g. Tesla T4, A10G, A100):

```bash
# 1. Install GPU dependencies
pip install -r training/requirements.txt
pip install -e sdk/

# 2. Configure credentials
export HF_TOKEN="your_hugging_face_write_token"
export BASE_MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"

# 3. Launch training with validation-based checkpointing
python training/train_qlora.py \
  --train-file training/data/train.jsonl \
  --val-file training/data/val.jsonl \
  --output-dir training/checkpoints \
  --max-steps 150 \
  --eval-steps 15 \
  --seed 42 \
  --push-to-hub
```

At completion, the pipeline logs `training_metadata.json` with the base model revision, training data SHA-256 hash, best validation step, and uploads the lightweight (~45 MB) LoRA adapter to Hugging Face.
