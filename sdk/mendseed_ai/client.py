"""
Async HTTP Client bridge for MendSeed AI.
Provides typed access to the remote 24/7 MendSeed AI FastAPI server,
validating all responses against strict Pydantic contracts.
"""

from typing import Any, Dict, List, Optional
import httpx

from .contracts import (
    CommunitySummary,
    DeveloperBrief,
    IssueAnalysis,
    IssueComparison,
    MergeProposal,
)


class MendSeedAIClient:
    """
    Zero-ML Client SDK for communicating with the MendSeed AI Cloud Microservice.
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
            "User-Agent": "MendSeed-AI-Client/0.1.0"
        }

    async def health_check(self) -> Dict[str, Any]:
        """Queries the /health endpoint to check server status and engine availability."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.endpoint_url}/health", headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    async def analyze_report(
        self,
        raw_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IssueAnalysis:
        """
        Submits raw community report text for structured extraction.
        Returns validated IssueAnalysis.
        """
        payload = {"text": raw_text, "metadata": metadata or {}}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/analyze",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return IssueAnalysis.model_validate(resp.json())

    async def compare_issues(
        self,
        report_a: str,
        report_b: str
    ) -> IssueComparison:
        """
        Compares two issue reports to establish duplicate/related relationship.
        Returns validated IssueComparison.
        """
        payload = {"report_a": report_a, "report_b": report_b}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/compare",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return IssueComparison.model_validate(resp.json())

    async def summarize_community(
        self,
        reports: List[str],
        category: Optional[str] = None
    ) -> CommunitySummary:
        """
        Synthesizes recurring pain points, themes, and disagreements across reports.
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

    async def generate_developer_brief(
        self,
        issue_title: str,
        reports: List[str]
    ) -> DeveloperBrief:
        """
        Generates an actionable developer brief for maintainers.
        Returns validated DeveloperBrief.
        """
        payload = {"title": issue_title, "reports": reports}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/developer-brief",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return DeveloperBrief.model_validate(resp.json())

    async def find_duplicates(
        self,
        raw_text: str,
        candidates: List[Dict[str, str]]
    ) -> MergeProposal:
        """
        Sends raw text and candidate canonical issues to evaluate potential duplicate merges.
        Returns validated MergeProposal.
        """
        payload = {"text": raw_text, "candidates": candidates}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint_url}/v1/propose-merge",
                headers=self.headers,
                json=payload
            )
            resp.raise_for_status()
            return MergeProposal.model_validate(resp.json())
