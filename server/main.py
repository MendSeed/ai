"""
MendSeed AI — 24/7 Global API Gateway & Public Grievance Observatory.

Production-ready FastAPI application designed for serverless deployment
(Vercel, Render, Fly.io, Cloud Run, Hugging Face Spaces).

MendSeed collects and structures real-world problems faced by people across diverse sectors
(civic infrastructure, agriculture, healthcare, education, logistics, governance, etc.)
and packages them for builders, researchers, and hackathon teams looking for genuine problems to solve.

Security & Architecture:
- Explicit MENDSEED_AI_SECRET_KEY authentication (HMAC constant-time verification)
- Clean inference provider interface (HuggingFace, custom, and mock providers)
- Strict differentiation between genuine model output and heuristic fallback (via response headers)
- Input bounds validation and rate-limiting hooks
- Zero browser-credentialed CORS; API keys must never be exposed to client-side JavaScript
- Canonical QuietFray v0.1 temporal evidence ranking engine
"""

from collections import defaultdict
import json
import logging
import os
import re
import secrets
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure sdk directory is in Python path for clean module resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Check sibling sdk or parent sdk
PARENT_DIR = os.path.dirname(CURRENT_DIR)
SDK_DIR = os.path.join(PARENT_DIR, "sdk")
if not os.path.exists(SDK_DIR):
    SDK_DIR = os.path.join(CURRENT_DIR, "sdk")
if os.path.exists(SDK_DIR) and SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mendseed_ai.contracts import (
    BuilderBrief,
    CanonicalClusterProposal,
    CommunitySummary,
    DeveloperBrief,
    GrievanceAnalysis,
    GrievanceComparison,
    GrievanceSector,
    IssueAnalysis,
    IssueComparison,
    MergeProposal,
    PLATFORM_CATEGORY_TO_SECTOR,
    map_platform_category_to_sector,
    map_sector_to_platform_category,
)
from mendseed_ai.quietfray import QuietFrayEngine, QuietFlayEngine
from server.provider import (
    BaseInferenceProvider,
    HuggingFaceInferenceProvider,
    MockInferenceProvider,
    ModelConfig,
    ProviderResult,
)

logger = logging.getLogger("mendseed_ai.gateway")
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI App
app = FastAPI(
    title="MendSeed AI — Public Grievance Observatory & Problem Discovery Gateway",
    description=(
        "Structures real-world citizen grievances across sectors (agriculture, civic, healthcare, "
        "education, mobility) into actionable problem briefs for builders and hackathon teams."
    ),
    version="0.2.0",
)

# ---------------------------------------------------------
# CORS Security Configuration
# ---------------------------------------------------------
# Never use allow_origins=["*"] with allow_credentials=True.
# The private MendSeed service key must never be exposed to browser JavaScript.
raw_cors = os.getenv("MENDSEED_CORS_ORIGINS", "")
if raw_cors.strip():
    allowed_origins = [o.strip() for o in raw_cors.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-MendSeed-Key"],
    )
else:
    # Safe default: Non-credentialed requests only
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-MendSeed-Key"],
    )

# ---------------------------------------------------------
# Rate Limiting In-Memory Hook
# ---------------------------------------------------------
RATE_LIMIT_PER_MINUTE = int(os.getenv("MENDSEED_RATE_LIMIT_PER_MINUTE", "120"))
_rate_limit_records = defaultdict(list)


def check_rate_limit(request: Request) -> None:
    """Simple sliding-window rate limiting hook."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    cutoff = now - 60.0

    # Prune old timestamps
    timestamps = [t for t in _rate_limit_records[client_ip] if t > cutoff]
    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please throttle your requests."
        )
    timestamps.append(now)
    _rate_limit_records[client_ip] = timestamps


# ---------------------------------------------------------
# Model & Provider Configuration
# ---------------------------------------------------------
model_config = ModelConfig.from_env()

if model_config.provider_type == "mock":
    inference_provider: BaseInferenceProvider = MockInferenceProvider(model_config)
else:
    inference_provider = HuggingFaceInferenceProvider(model_config)

# Singleton QuietFray Engine
quietfray_engine = QuietFrayEngine()
# Alias for backwards compatibility
quietflay_engine = quietfray_engine


def get_inference_provider() -> BaseInferenceProvider:
    """Dependency provider hook so test harnesses can cleanly inject mocks."""
    return getattr(app.state, "provider", inference_provider)


# ---------------------------------------------------------
# Security Header Dependency
# ---------------------------------------------------------
def verify_api_key(x_mendseed_key: Optional[str] = Header(None)) -> str:
    """
    Validates the X-MendSeed-Key header using constant-time comparison
    to prevent timing attacks.
    Requires explicitly configured MENDSEED_AI_SECRET_KEY.
    """
    configured_key = os.getenv("MENDSEED_AI_SECRET_KEY")
    if not configured_key or not configured_key.strip():
        # Fail safely: refuse protected calls if secret is unconfigured
        logger.error("Protected endpoint called, but MENDSEED_AI_SECRET_KEY is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: MENDSEED_AI_SECRET_KEY is unconfigured"
        )

    if not x_mendseed_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required header: X-MendSeed-Key"
        )

    if not secrets.compare_digest(x_mendseed_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized: Invalid MendSeed API key"
        )

    return x_mendseed_key


# ---------------------------------------------------------
# Request Payloads with Strict Bounds Validation
# ---------------------------------------------------------
class ReportPayload(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Raw problem report or citizen grievance"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Safe metadata (region, district, category tags)"
    )


class ComparePayload(BaseModel):
    report_a: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="First grievance text"
    )
    report_b: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Second grievance text"
    )


class CommunitySummaryPayload(BaseModel):
    reports: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of related public reports to synthesize"
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional sector or category label"
    )


class BuilderBriefPayload(BaseModel):
    title: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Canonical real-world problem title"
    )
    reports: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Associated citizen reports to extract ground realities from"
    )


class MergeProposalPayload(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="New incoming grievance report"
    )
    candidates: List[Dict[str, str]] = Field(
        ...,
        min_length=1,
        max_length=25,
        description="Candidate canonical problem statements with 'id', 'title', and 'summary'"
    )


class BranchingMassPayload(BaseModel):
    events: List[List[float]] = Field(
        ...,
        max_length=5000,
        description="List of [timestamp_in_days, quality_weight_q] tuples"
    )
    current_t: float = Field(..., description="Current evaluation timestamp in days")


class StreamingUpdatePayload(BaseModel):
    current_state: Dict[str, float] = Field(..., description="Existing issue state {x_fast, x_slow, log_lr, last_t}")
    event_time_days: float = Field(..., description="Timestamp of the new event in days")
    event_mark: float = Field(..., ge=0.0, le=1.0, description="Quality mark q_j in [0, 1]")
    is_report: bool = Field(default=True, description="True for problem report, False for upvote/downvote")


# ---------------------------------------------------------
# Fallback Heuristics
# ---------------------------------------------------------
def infer_sector_from_text(text: str) -> GrievanceSector:
    """Heuristic domain classifier for fallback processing."""
    t = text.lower()
    if any(w in t for w in ["farm", "crop", "seed", "harvest", "mandi", "fertilizer", "soil", "farmer", "irrigation", "paddy", "wheat", "tomato", "borewell"]):
        return "Agriculture & Rural Development"
    if any(w in t for w in ["water", "pipe", "sewage", "road", "pothole", "streetlight", "drain", "bridge", "pavement", "electricity", "transformer"]):
        return "Civic & Public Infrastructure"
    if any(w in t for w in ["hospital", "doctor", "health", "antivenom", "clinic", "medicine", "nurse", "ambulance", "vaccine", "sanitation", "infection"]):
        return "Healthcare & Sanitation"
    if any(w in t for w in ["school", "college", "teacher", "student", "class", "exam", "education", "book", "fees", "degree", "classroom"]):
        return "Education & Skilling"
    if any(w in t for w in ["bus", "train", "metro", "auto", "traffic", "commute", "fare", "transport", "station", "highway"]):
        return "Transportation & Mobility"
    if any(w in t for w in ["garbage", "waste", "pollution", "plastic", "smog", "air", "river", "lake", "forest"]):
        return "Environment & Waste Management"
    if any(w in t for w in ["pension", "ration", "aadhaar", "bribe", "panchayat", "collector", "official", "certificate", "scheme"]):
        return "Public Services & Local Governance"
    return "Livelihoods & Daily Life"


def set_inference_headers(response: Response, is_fallback: bool, status_desc: str = "success") -> None:
    """
    Sets documented response headers to distinguish neural inference from fallback.
    Never misrepresent a heuristic output as a neural-model result.
    """
    if is_fallback:
        response.headers["X-MendSeed-Inference-Source"] = "heuristic-fallback"
        response.headers["X-MendSeed-Fallback"] = "true"
        response.headers["X-Inference-Fallback"] = "true"
        response.headers["X-MendSeed-Fallback-Reason"] = status_desc
    else:
        response.headers["X-MendSeed-Inference-Source"] = "model-inference"
        response.headers["X-MendSeed-Fallback"] = "false"
        response.headers["X-Inference-Fallback"] = "false"


# ---------------------------------------------------------
# Public System Routes
# ---------------------------------------------------------
@app.get("/", tags=["System"])
def root_info():
    """Service index and operational capabilities."""
    return {
        "service": "MendSeed AI — Public Grievance Observatory",
        "mission": "Cataloging real-world community problems across sectors for hackathon teams and project builders",
        "version": "0.2.0",
        "engine": "QuietFray v0.1",
        "subcriticality_guarantee": True,
        "fast_decay_minutes": round(quietfray_engine.tau_f * 24 * 60, 1),
        "slow_decay_days": quietfray_engine.tau_s,
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["System"])
def health_check():
    """
    Public liveness and readiness probe.
    Does NOT leak secret configuration details or API token states.
    """
    cfg = model_config
    valid, reason = cfg.validate_configuration()
    return {
        "status": "healthy",
        "service": "MendSeed AI Gateway",
        "engine": "QuietFray v0.1",
        "provider_type": cfg.provider_type,
        "model_configured": valid,
        "gateway_ready": True
    }


# ---------------------------------------------------------
# Protected Inference Routes
# ---------------------------------------------------------
@app.post(
    "/v1/analyze",
    response_model=GrievanceAnalysis,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    tags=["Grievance Intelligence"]
)
async def analyze_report(
    payload: ReportPayload,
    response: Response,
    provider: BaseInferenceProvider = Depends(get_inference_provider)
):
    """
    Extracts structured GrievanceAnalysis from raw citizen problem report.
    Returns model extraction or documented fallback.
    """
    system_prompt = (
        "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. "
        "Extract structured data from raw citizen reports across diverse sectors. "
        "Return ONLY valid JSON matching the GrievanceAnalysis schema: "
        "category ('Civic & Public Infrastructure' | 'Agriculture & Rural Development' | "
        "'Healthcare & Sanitation' | 'Education & Skilling' | 'Environment & Waste Management' | "
        "'Transportation & Mobility' | 'Public Services & Local Governance' | 'Livelihoods & Daily Life' | 'Other Grievance'), "
        "normalized_title, summary, affected_sector, location_or_demographic, "
        "observed_friction, current_workarounds, missing_information, confidence."
    )
    user_prompt = f"Process citizen grievance:\n{payload.text}"

    result = await provider.generate(system_prompt, user_prompt)
    if result.success and result.text:
        try:
            parsed = json.loads(result.text)
            validated = GrievanceAnalysis.model_validate(parsed)
            if not validated.platform_category:
                validated.platform_category = map_sector_to_platform_category(validated.category)
            set_inference_headers(response, is_fallback=False)
            return validated
        except Exception as e:
            logger.warning(f"Model output parsing failed: {e}. Falling back to domain heuristic.")

    # Fallback path
    set_inference_headers(response, is_fallback=True, status_desc=result.status)
    sector = infer_sector_from_text(payload.text)
    first_sentence = payload.text.split(".")[0].strip()

    return GrievanceAnalysis(
        category=sector,
        platform_category=map_sector_to_platform_category(sector),
        normalized_title=first_sentence[:80] if len(first_sentence) > 10 else "Public Grievance Ingested",
        summary=payload.text[:220].strip(),
        affected_sector=sector,
        location_or_demographic=payload.metadata.get("region", "Local Community"),
        observed_friction=payload.text[:150],
        current_workarounds="Ad-hoc coping mechanisms reported by affected citizens",
        missing_information=["Specific administrative ward / block", "Seasonal variation in problem"],
        confidence=0.70  # Conservative confidence for heuristic fallback
    )


@app.post(
    "/v1/compare",
    response_model=GrievanceComparison,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    tags=["Grievance Intelligence"]
)
async def compare_issues(
    payload: ComparePayload,
    response: Response,
    provider: BaseInferenceProvider = Depends(get_inference_provider)
):
    """
    Compares two problem reports to evaluate if they share systemic causes or are distinct.
    """
    system_prompt = (
        "You are MendSeed AI. Compare two community problem reports to identify shared systemic causes. "
        "Return ONLY valid JSON matching GrievanceComparison schema: "
        "relationship ('Same underlying problem' | 'Related problem in same domain' | 'Distinct problem' | 'Insufficient information'), "
        "evidence, matching_score, meaningful_differences, recommended_action."
    )
    user_prompt = f"Grievance A:\n{payload.report_a}\n\nGrievance B:\n{payload.report_b}"

    result = await provider.generate(system_prompt, user_prompt)
    if result.success and result.text:
        try:
            parsed = json.loads(result.text)
            validated = GrievanceComparison.model_validate(parsed)
            set_inference_headers(response, is_fallback=False)
            return validated
        except Exception as e:
            logger.warning(f"Comparison parsing failed: {e}")

    # Fallback path
    set_inference_headers(response, is_fallback=True, status_desc=result.status)
    words_a = set(re.findall(r"\w+", payload.report_a.lower()))
    words_b = set(re.findall(r"\w+", payload.report_b.lower()))
    overlap = len(words_a & words_b)
    jaccard = overlap / max(1, len(words_a | words_b))

    rel = "Same underlying problem" if jaccard > 0.60 else ("Related problem in same domain" if jaccard > 0.22 else "Distinct problem")
    return GrievanceComparison(
        relationship=rel,
        evidence=[f"Vocabulary overlap index: {jaccard:.2f}"],
        matching_score=round(min(1.0, jaccard * 1.5), 2),
        meaningful_differences=["Evaluated via fallback surface vocabulary overlap"],
        recommended_action="cluster_under_canonical" if rel != "Distinct problem" else "keep_distinct"
    )


@app.post(
    "/v1/community-summary",
    response_model=CommunitySummary,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    tags=["Grievance Intelligence"]
)
async def summarize_community(
    payload: CommunitySummaryPayload,
    response: Response,
    provider: BaseInferenceProvider = Depends(get_inference_provider)
):
    """
    Synthesizes recurring grievances across multiple citizen reports in a region or sector.
    """
    # Category-aware synthesis
    category_ctx = ""
    if payload.category:
        norm_cat = map_platform_category_to_sector(payload.category) if payload.category in PLATFORM_CATEGORY_TO_SECTOR else payload.category
        category_ctx = f" Sector/Category Context: {norm_cat}."
    else:
        norm_cat = None

    system_prompt = (
        f"You are MendSeed AI. Synthesize recurring community grievances across multiple reports.{category_ctx} "
        "Return ONLY valid JSON matching CommunitySummary schema: "
        "main_problems (list), recurring_themes (list), disagreements (list), evidence_backed_feedback (list)."
    )
    user_prompt = "Citizen Reports:\n" + "\n---\n".join(payload.reports[:10])

    result = await provider.generate(system_prompt, user_prompt)
    if result.success and result.text:
        try:
            parsed = json.loads(result.text)
            validated = CommunitySummary.model_validate(parsed)
            set_inference_headers(response, is_fallback=False)
            return validated
        except Exception as e:
            logger.warning(f"Community summary parsing failed: {e}")

    # Fallback path
    set_inference_headers(response, is_fallback=True, status_desc=result.status)
    theme_lead = f"Service delivery bottlenecks in {norm_cat}" if norm_cat else "Service delivery bottlenecks"
    return CommunitySummary(
        main_problems=[f"Widespread grievance across {len(payload.reports)} citizen submissions: {payload.reports[0][:70]}..."],
        recurring_themes=[theme_lead, "Infrastructure fragility", "Need for systemic intervention"],
        disagreements=[],
        evidence_backed_feedback=[r[:120] for r in payload.reports[:3]]
    )


@app.post(
    "/v1/builder-brief",
    response_model=BuilderBrief,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    tags=["Builder & Hackathon Discovery"]
)
async def generate_builder_brief(
    payload: BuilderBriefPayload,
    response: Response,
    provider: BaseInferenceProvider = Depends(get_inference_provider)
):
    """
    Generates an actionable problem brief for project creators, researchers,
    and hackathon participants looking for real-world problems to solve.
    """
    system_prompt = (
        "You are MendSeed AI. Generate an actionable BuilderBrief for hackathon teams, researchers, and project creators "
        "looking for real-world problems to solve. Return ONLY valid JSON matching BuilderBrief schema: "
        "problem_statement, target_beneficiaries, current_ground_reality, impact, solution_opportunities (list), unresolved_questions (list)."
    )
    user_prompt = f"Problem Title: {payload.title}\n\nGround Evidence:\n" + "\n---\n".join(payload.reports[:5])

    result = await provider.generate(system_prompt, user_prompt)
    if result.success and result.text:
        try:
            parsed = json.loads(result.text)
            validated = BuilderBrief.model_validate(parsed)
            set_inference_headers(response, is_fallback=False)
            return validated
        except Exception as e:
            logger.warning(f"Builder brief parsing failed: {e}")

    # Fallback path
    set_inference_headers(response, is_fallback=True, status_desc=result.status)
    return BuilderBrief(
        problem_statement=f"Unresolved real-world friction: {payload.title}",
        target_beneficiaries="Local community members and impacted citizens",
        current_ground_reality=payload.reports[0][:180] if payload.reports else "Current solutions fail to address localized constraints.",
        impact="Causes recurring economic, time, or social hardship for affected households.",
        solution_opportunities=[
            "Design a low-cost, decentralized community monitoring prototype",
            "Build an offline-first coordination tool for affected citizens",
            "Create an open data visualization dashboard for local administrative accountability"
        ],
        unresolved_questions=[
            "What low-bandwidth or offline constraints must the prototype accommodate?",
            "What existing local organizations can pilot an initial prototype?"
        ]
    )


# Backward compatibility alias
@app.post(
    "/v1/developer-brief",
    response_model=BuilderBrief,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    include_in_schema=False
)
async def generate_developer_brief_alias(
    payload: BuilderBriefPayload,
    response: Response,
    provider: BaseInferenceProvider = Depends(get_inference_provider)
):
    return await generate_builder_brief(payload, response, provider)


@app.post(
    "/v1/propose-merge",
    response_model=CanonicalClusterProposal,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    tags=["Grievance Intelligence"]
)
async def propose_merge(
    payload: MergeProposalPayload,
    response: Response,
    provider: BaseInferenceProvider = Depends(get_inference_provider)
):
    """
    Evaluates whether an incoming citizen grievance clusters under a canonical problem umbrella.
    Always enforces human_review_required=True.
    """
    if not payload.candidates:
        set_inference_headers(response, is_fallback=True, status_desc="no_candidates")
        return CanonicalClusterProposal(
            candidate_canonical_problem_id="",
            proposed_grievance_links=[],
            supporting_evidence="No canonical problem candidates provided for comparison.",
            uncertainty_level="High",
            human_review_required=True
        )

    system_prompt = (
        "You are MendSeed AI. Evaluate whether incoming citizen grievances cluster under a canonical real-world problem statement. "
        "Return ONLY valid JSON matching CanonicalClusterProposal schema: "
        "candidate_canonical_problem_id, proposed_grievance_links, supporting_evidence, uncertainty_level, human_review_required."
    )
    user_prompt = f"Incoming Grievance: {payload.text}\nCanonical Candidates: {json.dumps(payload.candidates)}"

    result = await provider.generate(system_prompt, user_prompt)
    if result.success and result.text:
        try:
            parsed = json.loads(result.text)
            # Enforce human review unconditionally
            parsed["human_review_required"] = True
            validated = CanonicalClusterProposal.model_validate(parsed)
            set_inference_headers(response, is_fallback=False)
            return validated
        except Exception as e:
            logger.warning(f"Merge proposal parsing failed: {e}")

    # Fallback path
    set_inference_headers(response, is_fallback=True, status_desc=result.status)
    best_candidate = payload.candidates[0]
    cand_id = best_candidate.get("id", "PROB-001")
    cand_title = best_candidate.get("title", "Existing Problem")

    return CanonicalClusterProposal(
        candidate_canonical_problem_id=cand_id,
        proposed_grievance_links=[cand_id],
        supporting_evidence=f"Ground symptoms align with canonical problem statement: '{cand_title}'.",
        uncertainty_level="Medium",
        human_review_required=True
    )


# ---------------------------------------------------------
# QuietFray v0.1 Mathematical Ranking Endpoints
# ---------------------------------------------------------
@app.post(
    "/v1/quietfray/evaluate",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    tags=["QuietFray Engine"]
)
def evaluate_quietfray_branching_mass(payload: BranchingMassPayload):
    """
    Computes dual-decay temporal branching mass (tau_f = 0.02 days, tau_s = 3.0 days).
    Filters out short-term viral spikes while identifying persistent unresolved friction.
    """
    events = [(float(e[0]), float(e[1])) for e in payload.events]
    result = quietfray_engine.evaluate_branching_mass(events, payload.current_t)
    return result


@app.post(
    "/v1/quietfray/streaming-update",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
    tags=["QuietFray Engine"]
)
def quietfray_streaming_update(payload: StreamingUpdatePayload):
    """
    Performs an O(1) online state recurrence update for a newly arrived vote or report
    without scanning historical records.
    """
    updated = quietfray_engine.update_streaming_state(
        current_state=payload.current_state,
        event_time_days=payload.event_time_days,
        event_mark=payload.event_mark,
        is_report=payload.is_report
    )
    return updated


# Backward compatibility aliases for QuietFlay routes
@app.post("/v1/quietflay/evaluate", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)], include_in_schema=False)
def evaluate_quietflay_alias(payload: BranchingMassPayload):
    return evaluate_quietfray_branching_mass(payload)


@app.post("/v1/quietflay/streaming-update", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)], include_in_schema=False)
def quietflay_update_alias(payload: StreamingUpdatePayload):
    return quietfray_streaming_update(payload)
