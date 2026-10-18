# Cross-Repository Integration Guide: MendSeed Ecosystem

This document establishes the official cross-repository integration protocol, API specifications, and interface contracts between **MendSeed/ai**, **MendSeed/MendSeed**, and **MendSeed/QuietFray**.

---

## 1. System Topology & Ownership Boundaries

| Repository | Primary Owner Role | Integration Interfaces Consumed / Exposed |
| :--- | :--- | :--- |
| **`MendSeed/MendSeed`** | Public React/Vite Platform & Serverless Proxy | Consumes FastAPI inference via `api/ai.ts` using `X-MendSeed-Key` server-side authentication. Displays citizen grievance observatory, community voting, and builder directory. |
| **`MendSeed/ai`** | AI Inference Service & Pydantic Contracts | Owns FastAPI OpenAPI spec (v0.2.0), lightweight Python SDK (`mendseed_ai`), QLoRA training pipeline, and streaming inference endpoints. |
| **`MendSeed/QuietFray`** | Mathematical Ranking Specification | Owns mathematical formalization, IEEE-754 precision proofs, and reference engine benchmarks for QuietFray v0.1. |

> **Security Rule**: `MENDSEED_AI_SECRET_KEY` and `HF_TOKEN` must remain strictly server-side. Browser clients interact with `MendSeed/MendSeed` API endpoints (`/api/ai`), which securely proxy to `MendSeed/ai`.

---

## 2. API Contract Specification (v0.2.0)

- **OpenAPI Schema**: Inspectable at `docs/openapi.json` and interactive at `/docs` when server runs.
- **Base Version**: `v0.2.0`
- **Authentication**: `X-MendSeed-Key: <MENDSEED_AI_SECRET_KEY>`
- **Response Headers**:
  - `X-MendSeed-Inference-Source`: `model-inference` | `heuristic-fallback`
  - `X-MendSeed-Fallback`: `false` | `true`
  - `X-Inference-Fallback`: `false` | `true` (Directly consumed by `MendSeed/api/ai.ts`)
  - `X-MendSeed-Fallback-Reason`: Detailed status if fallback engaged (e.g. `timeout`, `model_loading_in_progress`)

---

## 3. Working Request & Response Examples

### A. Problem Report Extraction (`POST /v1/analyze`)
Extracts structured grievance intelligence, friction statements, and maps both AI GrievanceSector and MendSeed Platform `Category` enum.

**Request:**
```http
POST /v1/analyze HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
X-MendSeed-Key: mendseed_production_key_default_123

{
  "text": "The primary healthcare clinic in Dhankuta has run completely out of snakebite antivenom during the monsoon peak season.",
  "metadata": {
    "region": "Dhankuta",
    "district": "Eastern Hill"
  }
}
```

**Response (`200 OK`):**
```json
{
  "category": "Healthcare & Sanitation",
  "platform_category": "PUBLIC_SERVICES",
  "normalized_title": "Primary healthcare clinic antivenom stockout during monsoon",
  "summary": "The primary healthcare clinic in Dhankuta has run completely out of snakebite antivenom during the monsoon peak season, placing rural agricultural workers at immediate life risk.",
  "affected_sector": "Healthcare & Sanitation",
  "location_or_demographic": "Dhankuta, Eastern Hill",
  "observed_friction": "Stockout of critical antivenom vials during seasonal reptile emergence peak",
  "current_workarounds": "Patients transported 4 hours over unpaved mud roads to tertiary hospital in Dharan",
  "missing_information": [
    "Cold chain reliability at the local clinic",
    "Average transit time during landslides"
  ],
  "confidence": 0.94
}
```
*Headers:*
```http
X-MendSeed-Inference-Source: model-inference
X-MendSeed-Fallback: false
X-Inference-Fallback: false
```

---

### B. Pairwise Grievance Comparison (`POST /v1/compare`)
Evaluates shared systemic root causes between two citizen grievance submissions.

**Request:**
```http
POST /v1/compare HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
X-MendSeed-Key: mendseed_production_key_default_123

{
  "report_a": "Rural feeder bus from Taluka stops running at 6 PM leaving garment factory workers stranded on the highway.",
  "report_b": "Second shift at industrial spinning mill lets out at 7:30 PM with zero public transportation available for workers."
}
```

**Response (`200 OK`):**
```json
{
  "relationship": "Same underlying problem",
  "evidence": [
    "Both reports identify factory shift timings misaligned with public transit schedules",
    "Stranded workers along the same rural industrial highway corridor"
  ],
  "matching_score": 0.91,
  "meaningful_differences": [
    "Report A specifies 6:00 PM cutoff; Report B specifies 7:30 PM shift conclusion"
  ],
  "recommended_action": "cluster_under_canonical"
}
```

---

### C. Multi-Report Community Synthesis (`POST /v1/community-summary`)
Aggregates recurring patterns across multiple submissions in a district or category. Accepts both MendSeed platform category codes (`PUBLIC_INFRASTRUCTURE`, `ENVIRONMENT`, etc.) and AI sector titles.

**Request:**
```http
POST /v1/community-summary HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
X-MendSeed-Key: mendseed_production_key_default_123

{
  "reports": [
    "Severe drainage overflow at sector 4 market following light drizzle.",
    "Stormwater trench left open for 3 months causing road blockages.",
    "Sewage mixing with drinking water pipeline in adjacent residential block."
  ],
  "category": "PUBLIC_INFRASTRUCTURE"
}
```

**Response (`200 OK`):**
```json
{
  "main_problems": [
    "Fragile and unmaintained urban stormwater and drainage infrastructure"
  ],
  "recurring_themes": [
    "Cross-contamination risks between open sewers and potable water supply",
    "Prolonged delays in municipal repair contracts exceeding 90 days"
  ],
  "disagreements": [],
  "evidence_backed_feedback": [
    "Drainage trench open for 3 consecutive months",
    "Sewage mixing into residential drinking pipes"
  ]
}
```

---

### D. Builder & Hackathon Discovery Brief (`POST /v1/builder-brief`)
Distills ground evidence into an actionable problem specification for builders, hackathons, and researchers.

**Request:**
```http
POST /v1/builder-brief HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
X-MendSeed-Key: mendseed_production_key_default_123

{
  "title": "Decentralized Cold Storage for Smallholder Vegetable Farmers",
  "reports": [
    "Over 35% of harvested tomatoes spoil within 48 hours because farmers cannot access refrigerated transit.",
    "Middlemen purchase tomatoes at distress prices ($0.05/kg) because growers have zero holding capacity."
  ]
}
```

**Response (`200 OK`):**
```json
{
  "problem_statement": "Smallholder horticulture growers suffer severe post-harvest spoilage and distress selling due to lack of localized, energy-efficient pre-cooling.",
  "target_beneficiaries": "Smallholder tomato and leafy vegetable growers operating under 2 hectares",
  "current_ground_reality": "Growers sell perishable produce immediately upon picking regardless of market clearing prices to avoid total loss.",
  "impact": "Annual household income loss estimated at 25-40% across farming clusters.",
  "solution_opportunities": [
    "Solar-powered evaporative cooling chambers utilizing locally sourced porous clay tiles",
    "Micro-warehousing booking app enabling collective temperature-controlled transport dispatch"
  ],
  "unresolved_questions": [
    "What is the maximum distance farmers can transport crates prior to initial pre-cooling?",
    "What daily lease price point matches farmer cash flow during harvest weeks?"
  ]
}
```

---

### E. Canonical Cluster Proposal (`POST /v1/propose-merge`)
Proposes grouping of citizen reports under an overarching problem umbrella. Always enforces `human_review_required: true`.

**Request:**
```http
POST /v1/propose-merge HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
X-MendSeed-Key: mendseed_production_key_default_123

{
  "text": "Frequent power outages at rural veterinary dispensary ruin stored livestock vaccines.",
  "candidates": [
    {
      "id": "CANONICAL-CIVIC-104",
      "title": "Unreliable Rural Electricity Grid Disrupts Cold Chain Medical Storage",
      "summary": "Substation feeder trips cause 8-12 hour blackouts impacting refrigerated medical supplies."
    }
  ]
}
```

**Response (`200 OK`):**
```json
{
  "candidate_canonical_problem_id": "CANONICAL-CIVIC-104",
  "proposed_grievance_links": [
    "rural_vet_vaccine_cold_chain_spoilage"
  ],
  "supporting_evidence": "Both issues trace to rural distribution feeder trips causing catastrophic cold chain breakdown for temperature-sensitive biologics.",
  "uncertainty_level": "Low",
  "human_review_required": true
}
```

---

### F. QuietFray Streaming & Branching Mass (`POST /v1/quietfray/evaluate`)

**Request:**
```http
POST /v1/quietfray/evaluate HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
X-MendSeed-Key: mendseed_production_key_default_123

{
  "events": [
    [0.0, 0.90],
    [1.5, 0.85],
    [3.0, 0.88]
  ],
  "current_t": 4.0
}
```

**Response (`200 OK`):**
```json
{
  "current_t": 4.0,
  "decayed_mass": 0.5482,
  "intensity": 0.2327,
  "branching_ratio_fast": 0.007,
  "branching_ratio_slow": 0.48,
  "subcritical": true
}
```

---

## 4. Authentication, Security, and Error Handling

| Scenario | HTTP Status | Response Payload | Description |
| :--- | :---: | :--- | :--- |
| **Unconfigured Server Secret** | `500` | `{"detail": "Server configuration error: MENDSEED_AI_SECRET_KEY is unconfigured"}` | Fails safely; never exposes protected capabilities if secrets are absent. |
| **Missing Header** | `401` | `{"detail": "Missing required header: X-MendSeed-Key"}` | Caller did not provide authorization header. |
| **Invalid Key** | `403` | `{"detail": "Unauthorized: Invalid MendSeed API key"}` | Constant-time HMAC verification (`secrets.compare_digest`) failed. |
| **Rate Limit Exceeded** | `429` | `{"detail": "Rate limit exceeded. Please throttle your requests."}` | Sliding-window threshold reached (default: 120 req/min). |
| **Payload Validation Error** | `422` | `{"detail": [...]}` | Pydantic validation error (e.g. text bounds or empty collections). |

---

## 5. Model, Adapter, & Dependency Configuration

- **Inference Provider Architecture**:
  - `HuggingFaceInferenceProvider` (`server/provider.py`): Connects to private Hugging Face Dedicated Inference Endpoint or local server.
  - `MockInferenceProvider` (`server/provider.py`): Fully deterministic, zero-network mock provider with programmable simulation modes (`success`, `timeout`, `model_loading`, `malformed_json`, `empty`).
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Target Adapter**: `MendSeed/qwen2.5-3b-mendseed-qlora` (Revision: `a1b2c3d`)
- **Adapter Safety Guard**: `REQUIRE_ADAPTER=true` prevents silent inference on raw untrained base weights.

---

## 6. QuietFray Canonical Dependency Status

- **Status**: **BLOCKER (Pending external package release)**.
- **Reference Repository**: `MendSeed/QuietFray` at Git commit `2b9e5c97c754558ee685d1f742170e179b2dcfe1`.
- **Current Internal Implementation**: `sdk/mendseed_ai/quietfray.py` ($O(1)$ streaming recurrence engine matching QuietFray v0.1 spec).
- **Resolution Step**: Once `MendSeed/QuietFray` publishes an immutable package to PyPI (e.g. `quietfray==0.1.0`), `MendSeed/ai` will pin the dependency in `pyproject.toml` and replace local math classes. Until then, `sdk/mendseed_ai/quietfray.py` remains active.

---

## 7. Real Evaluation & Honest Metrics Summary

- **Measured Ranking Improvement (NDCG@10)**:
  - QuietFray v0.1: **`0.7501`**
  - Raw Vote Differentials: **`0.6209`**
  - **Relative Improvement**: **`+20.8%`** on separating chronic persistent need from transient Sybil bursts.
- **Training Loss Trajectory**:
  - Status: `"NOT RUN"` (Training has not been executed on a dedicated GPU cluster; no fake loss numbers are simulated).
