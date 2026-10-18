"""
Unit Tests for MendSeed AI FastAPI Endpoints, Input Bounds, and Response Headers.

Tests:
1. All 5 core endpoints return strictly valid Pydantic models:
   - /v1/analyze -> GrievanceAnalysis
   - /v1/compare -> GrievanceComparison
   - /v1/community-summary -> CommunitySummary
   - /v1/builder-brief -> BuilderBrief
   - /v1/propose-merge -> CanonicalClusterProposal
2. Input length and collection size boundary validations (max length, min length, max items).
3. Response headers strictly distinguish genuine neural model output from heuristic fallback:
   - Success: X-MendSeed-Inference-Source: model-inference, X-MendSeed-Fallback: false
   - Fallback: X-MendSeed-Inference-Source: heuristic-fallback, X-MendSeed-Fallback: true
4. Backward compatibility aliases:
   - /v1/developer-brief
   - /v1/quietflay/evaluate
   - /v1/quietflay/streaming-update
"""

import os
import pytest
from fastapi.testclient import TestClient

from server.main import app, get_inference_provider
from server.provider import MockInferenceProvider

TEST_SECRET = "test_mendseed_secret_key_abcdef123456"
HEADERS = {"X-MendSeed-Key": TEST_SECRET}


@pytest.fixture(autouse=True)
def setup_test_app():
    original = os.environ.get("MENDSEED_AI_SECRET_KEY")
    os.environ["MENDSEED_AI_SECRET_KEY"] = TEST_SECRET
    mock_provider = MockInferenceProvider()
    app.dependency_overrides[get_inference_provider] = lambda: mock_provider
    yield mock_provider
    app.dependency_overrides.clear()
    if original is not None:
        os.environ["MENDSEED_AI_SECRET_KEY"] = original
    else:
        os.environ.pop("MENDSEED_AI_SECRET_KEY", None)


def test_analyze_endpoint_model_success():
    client = TestClient(app)
    payload = {
        "text": "The primary healthcare clinic in Dhankuta has no antivenom during monsoon.",
        "metadata": {"region": "Dhankuta"}
    }
    resp = client.post("/v1/analyze", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["category"] in [
        "Healthcare & Sanitation",
        "Civic & Public Infrastructure",
        "Agriculture & Rural Development"
    ]
    assert data["confidence"] > 0.0
    assert resp.headers.get("X-MendSeed-Inference-Source") == "model-inference"
    assert resp.headers.get("X-MendSeed-Fallback") == "false"
    assert resp.headers.get("X-Inference-Fallback") == "false"
    assert data.get("platform_category") in [
        "PUBLIC_INFRASTRUCTURE",
        "PUBLIC_SERVICES",
        "ENVIRONMENT",
        "EDUCATION",
        "TRANSPORTATION",
        "EVERYDAY_LIFE",
        "OTHER",
    ]


def test_analyze_endpoint_fallback_headers(setup_test_app):
    """
    When the provider fails, the endpoint returns a valid schema but sets
    explicit fallback headers so clients can distinguish it from genuine model reasoning.
    """
    mock_provider = setup_test_app
    mock_provider.simulate_mode = "timeout"

    client = TestClient(app)
    payload = {
        "text": "Borewell ran dry in village farm, tomato harvest spoiled completely.",
        "metadata": {"region": "Kolar"}
    }
    resp = client.post("/v1/analyze", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["category"] == "Agriculture & Rural Development"
    assert data["platform_category"] == "ENVIRONMENT"
    assert resp.headers.get("X-MendSeed-Inference-Source") == "heuristic-fallback"
    assert resp.headers.get("X-MendSeed-Fallback") == "true"
    assert resp.headers.get("X-Inference-Fallback") == "true"
    assert "timeout" in resp.headers.get("X-MendSeed-Fallback-Reason", "").lower()


def test_compare_endpoint():
    client = TestClient(app)
    payload = {
        "report_a": "Rural feeder bus from Taluka stops running at 6 PM leaving garment workers stranded.",
        "report_b": "Textile factory shift finishes at 7:30 PM with zero public transportation available."
    }
    resp = client.post("/v1/compare", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["relationship"] in [
        "Same underlying problem",
        "Related problem in same domain",
        "Distinct problem",
        "Insufficient information"
    ]
    assert isinstance(data["evidence"], list)
    assert 0.0 <= data["matching_score"] <= 1.0


def test_community_summary_endpoint():
    client = TestClient(app)
    payload = {
        "reports": [
            "Potholes on main market road caused two accidents.",
            "Water stagnation after rain makes market lane unusable.",
            "Drainage trench left open for 2 weeks."
        ],
        "category": "Civic & Public Infrastructure"
    }
    resp = client.post("/v1/community-summary", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert isinstance(data["main_problems"], list)
    assert isinstance(data["recurring_themes"], list)
    assert isinstance(data["evidence_backed_feedback"], list)


def test_community_summary_with_platform_category_enum():
    client = TestClient(app)
    payload = {
        "reports": [
            "Potholes on main market road caused two accidents.",
            "Water stagnation after rain makes market lane unusable."
        ],
        "category": "PUBLIC_INFRASTRUCTURE"
    }
    resp = client.post("/v1/community-summary", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["main_problems"], list)
    assert isinstance(data["recurring_themes"], list)
    assert resp.headers.get("X-Inference-Fallback") in ["true", "false"]


def test_builder_brief_endpoint():
    client = TestClient(app)
    payload = {
        "title": "Decentralized Cold Storage for Smallholder Farmers",
        "reports": [
            "Tomato harvest lost due to lack of localized cooling.",
            "Mandi transit distance causes distress sales."
        ]
    }
    resp = client.post("/v1/builder-brief", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "problem_statement" in data
    assert "target_beneficiaries" in data
    assert isinstance(data["solution_opportunities"], list)
    assert isinstance(data["unresolved_questions"], list)


def test_propose_merge_endpoint_enforces_human_review():
    client = TestClient(app)
    payload = {
        "text": "Tubewell is pumping saline fluoride water in Ward 8.",
        "candidates": [
            {"id": "PROB-001", "title": "Groundwater table contamination", "summary": "Fluoride in sub-district aquifer."}
        ]
    }
    resp = client.post("/v1/propose-merge", headers=HEADERS, json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["candidate_canonical_problem_id"] == "PROB-001"
    assert data["human_review_required"] is True  # Mandatory governance constraint


def test_input_bounds_validation():
    client = TestClient(app)

    # 1. Text length exceeding 5000 characters
    oversized_text = "A" * 5001
    resp = client.post("/v1/analyze", headers=HEADERS, json={"text": oversized_text})
    assert resp.status_code == 422

    # 2. Text too short (< 3 characters)
    resp = client.post("/v1/analyze", headers=HEADERS, json={"text": "hi"})
    assert resp.status_code == 422

    # 3. Community summary reports list empty
    resp = client.post("/v1/community-summary", headers=HEADERS, json={"reports": []})
    assert resp.status_code == 422

    # 4. Community summary reports exceeding 50 items
    resp = client.post(
        "/v1/community-summary",
        headers=HEADERS,
        json={"reports": [f"Report {i}" for i in range(51)]}
    )
    assert resp.status_code == 422


def test_backward_compatibility_aliases():
    client = TestClient(app)

    # 1. /v1/developer-brief alias
    resp_alias = client.post(
        "/v1/developer-brief",
        headers=HEADERS,
        json={"title": "Solar Dryer", "reports": ["Sample report"]}
    )
    assert resp_alias.status_code == 200
    assert "problem_statement" in resp_alias.json()

    # 2. /v1/quietflay/evaluate alias
    resp_qf = client.post(
        "/v1/quietflay/evaluate",
        headers=HEADERS,
        json={"events": [[1.0, 0.9], [2.0, 0.8]], "current_t": 3.0}
    )
    assert resp_qf.status_code == 200
    assert "composite_score" in resp_qf.json()
