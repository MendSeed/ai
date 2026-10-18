"""
Unit Tests for MendSeed AI Client SDK.

Tests:
1. MendSeedAIClient communicating with FastAPI server.
2. Deserialization of all contracts:
   - GrievanceAnalysis
   - GrievanceComparison
   - CommunitySummary
   - BuilderBrief
   - CanonicalClusterProposal
3. Remote QuietFray branching mass evaluation.
"""

import os
import pytest
from httpx import ASGITransport, AsyncClient

from mendseed_ai import (
    BuilderBrief,
    CanonicalClusterProposal,
    CommunitySummary,
    GrievanceAnalysis,
    GrievanceComparison,
    MendSeedAIClient,
)
from server.main import app, get_inference_provider
from server.provider import MockInferenceProvider

TEST_SECRET = "test_mendseed_secret_key_abcdef123456"


@pytest.fixture(autouse=True)
def setup_sdk_test():
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


@pytest.mark.asyncio
async def test_sdk_client_end_to_end(monkeypatch):
    """
    Patches httpx.AsyncClient inside the SDK client to route directly to ASGI app.
    """
    transport = ASGITransport(app=app)
    real_async_client = AsyncClient

    def custom_client(**kwargs):
        kwargs["transport"] = transport
        kwargs.pop("base_url", None)
        return real_async_client(base_url="http://testserver", **kwargs)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", custom_client)

    sdk = MendSeedAIClient(endpoint_url="http://testserver", api_key=TEST_SECRET)

    # 1. Health check
    health = await sdk.health_check()
    assert health["status"] == "healthy"

    # 2. Analyze grievance
    analysis = await sdk.analyze_grievance("Farmers face crop loss due to power cuts in rural feeder.")
    assert isinstance(analysis, GrievanceAnalysis)
    assert analysis.category in [
        "Agriculture & Rural Development",
        "Civic & Public Infrastructure",
        "Healthcare & Sanitation"
    ]

    # 3. Compare grievances
    comparison = await sdk.compare_grievances("Report 1 text", "Report 2 text")
    assert isinstance(comparison, GrievanceComparison)
    assert comparison.relationship in [
        "Same underlying problem",
        "Related problem in same domain",
        "Distinct problem",
        "Insufficient information"
    ]

    # 4. Community summary
    summary = await sdk.summarize_community(["Report 1", "Report 2"])
    assert isinstance(summary, CommunitySummary)
    assert len(summary.main_problems) > 0

    # 5. Builder brief
    brief = await sdk.generate_builder_brief("Decentralized Cold Storage", ["Report text"])
    assert isinstance(brief, BuilderBrief)
    assert len(brief.problem_statement) > 0

    # 6. Duplicates / Canonical clustering
    clustering = await sdk.find_duplicates(
        "Groundwater issue",
        [{"id": "PROB-01", "title": "Aquifer", "summary": "Salinity"}]
    )
    assert isinstance(clustering, CanonicalClusterProposal)
    assert clustering.human_review_required is True

    # 7. QuietFray remote branching
    qf_res = await sdk.evaluate_quietfray_branching([(1.0, 0.9), (2.0, 0.85)], current_t=2.5)
    assert "composite_score" in qf_res
    assert "fast_mass" in qf_res
    assert "slow_mass" in qf_res
