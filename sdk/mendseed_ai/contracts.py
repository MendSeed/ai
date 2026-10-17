"""
Pydantic contracts for MendSeed AI issue intelligence and governance.
These five schemas define the strict, typed boundaries between the AI inference layer
and the core MendSeed platform.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class IssueAnalysis(BaseModel):
    """
    Structured extraction of a raw community report.
    Separates what the author reported from what is inferred.
    """
    category: Literal["Bug", "Feature Request", "Inquiry"] = Field(
        description="Broad category of the grievance or request"
    )
    normalized_title: str = Field(
        description="Normalized, concise summary title of the issue"
    )
    summary: str = Field(
        description="One-sentence neutral summary of the core issue"
    )
    affected_area: str = Field(
        description="Application module, feature, or domain affected"
    )
    reproducible_details: Optional[str] = Field(
        default=None,
        description="Replication steps, device, or environment details if provided"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Key information absent from the report needed for resolution"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score in the structured extraction"
    )


class IssueComparison(BaseModel):
    """
    Comparison between two problem reports to evaluate relationship and similarity.
    """
    relationship: Literal[
        "Same underlying issue",
        "Related issue",
        "Distinct issue",
        "Insufficient information"
    ] = Field(
        description="Classified relationship between the new report and the candidate"
    )
    evidence: List[str] = Field(
        description="Specific symptoms, error codes, or user workflows shared between reports"
    )
    matching_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic similarity or compatibility score between reports"
    )
    meaningful_differences: List[str] = Field(
        default_factory=list,
        description="Differences in environment, triggers, or observed failure modes"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Proposed moderator or user action (e.g., 'link', 'merge', 'keep_separate')"
    )


class CommunitySummary(BaseModel):
    """
    Synthesized community feedback across a collection of related reports.
    """
    main_problems: List[str] = Field(
        description="Primary unresolved pain points reported across the group"
    )
    recurring_themes: List[str] = Field(
        description="Common workflows, conditions, or user personas encountering the problem"
    )
    disagreements: List[str] = Field(
        default_factory=list,
        description="Conflicting user reports, workaround disputes, or divergent opinions"
    )
    evidence_backed_feedback: List[str] = Field(
        description="Actionable takeaways directly grounded in actual quoted or referenced reports"
    )


class DeveloperBrief(BaseModel):
    """
    Technical issue brief formatted for maintainers and engineers.
    """
    problem_statement: str = Field(
        description="Concise description of the failure mode or requirement"
    )
    expected_behavior: str = Field(
        description="What the application was expected to do"
    )
    actual_behavior: str = Field(
        description="What the application actually did"
    )
    impact: str = Field(
        description="User friction, data risk, or operational consequence"
    )
    reproduction_steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step reproduction path based on reported facts"
    )
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="Open questions for maintainers before implementation begins"
    )


class MergeProposal(BaseModel):
    """
    Proposed duplicate linking between community reports.
    Consequential merges always require human moderator approval.
    """
    candidate_canonical_issue_id: str = Field(
        description="ID of the existing canonical issue being suggested as the parent"
    )
    proposed_duplicate_links: List[str] = Field(
        default_factory=list,
        description="List of report IDs suggested to link to the canonical issue"
    )
    supporting_evidence: str = Field(
        description="Detailed rationale and symptom matches supporting the proposed merge"
    )
    uncertainty_level: Literal["Low", "Medium", "High"] = Field(
        description="Estimated uncertainty level of the proposal"
    )
    human_review_required: bool = Field(
        default=True,
        description="Guardrail: human review is strictly mandatory before canonical record modification"
    )
