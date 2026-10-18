"""
Async HTTP Client bridge for MendSeed AI.
Provides typed access to the remote 24/7 MendSeed AI FastAPI server,
validating all responses against strict Pydantic contracts.
"""

from typing import Any, Dict, List, Optional, Tuple
import httpx

from .contracts import (
    BuilderBrief,
    CanonicalClusterProposal,
    CommunitySummary,
    DeveloperBrief,
    GrievanceAnalysis,
    GrievanceComparison,
    IssueAnalysis,
    IssueComparison,
    MergeProposal,
)


class MendSeedAIClient:
    """
    Zero-ML Client SDK for communicating with the MendSeed Public Grievance AI Microservice.
    Uses async HTTP requests with X-MendSeed-Key header authentication.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        timeout: float = 30.0
    ):
        """
        :param endpoint_url: Base URL of the MendSeed AI FastAPI service (e.g., https://ai.mendseed.org)
        :param api_key: Authentication secret key matching MENDSEED_AI_SECRET_KEY
        :param timeout: HTTP request timeout in seconds
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "X-MendSeed-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "MendSeed-AI-Client/0.2.0"
        }

    async def health_check(self) -> Dict[str, Any]:
        """Queries the /health endpoint to check server status and engine availability."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.endpoint_url}/health", headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    async def analyze_grievance(
        self,
        raw_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GrievanceAnalysis:
        """
        Submits raw citizen grievance text for structured extraction.
        Returns validated GrievanceAnalysis.
        """
        payload = {"text": raw_text, "metadata": metadata or {}}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/analyze",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return GrievanceAnalysis.model_validate(resp.json())

    # Backward compatibility alias
    analyze_report = analyze_grievance

    async def compare_grievances(
        self,
        report_a: str,
        report_b: str
    ) -> GrievanceComparison:
        """
        Compares two grievances to evaluate if they share root causes or are distinct.
        Returns validated GrievanceComparison.
        """
        payload = {"report_a": report_a, "report_b": report_b}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/compare",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return GrievanceComparison.model_validate(resp.json())

    # Backward compatibility alias
    compare_issues = compare_grievances

    async def summarize_community(
        self,
        reports: List[str],
        category: Optional[str] = None
    ) -> CommunitySummary:
        """
        Synthesizes recurring pain points, themes, and evidence across public reports.
        Returns validated CommunitySummary.
        """
        payload = {"reports": reports, "category": category}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/community-summary",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return CommunitySummary.model_validate(resp.json())

    async def generate_builder_brief(
        self,
        problem_title: str,
        reports: List[str]
    ) -> BuilderBrief:
        """
        Generates an actionable problem brief for project builders and hackathons.
        Returns validated BuilderBrief.
        """
        payload = {"title": problem_title, "reports": reports}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/builder-brief",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return BuilderBrief.model_validate(resp.json())

    # Backward compatibility alias
    generate_developer_brief = generate_builder_brief

    async def find_duplicates(
        self,
        raw_text: str,
        candidates: List[Dict[str, str]]
    ) -> CanonicalClusterProposal:
        """
        Evaluates whether a new grievance clusters with existing canonical problem statements.
        Returns validated CanonicalClusterProposal.
        """
        payload = {"text": raw_text, "candidates": candidates}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/propose-merge",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return CanonicalClusterProposal.model_validate(resp.json())

    async def evaluate_quietfray_branching(
        self,
        events: List[Tuple[float, float]],
        current_t: float
    ) -> Dict[str, float]:
        """Queries remote gateway for QuietFray branching mass evaluation."""
        payload = {"events": [[float(t), float(q)] for t, q in events], "current_t": current_t}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/quietfray/evaluate",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return resp.json()
