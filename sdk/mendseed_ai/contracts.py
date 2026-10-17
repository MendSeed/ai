"""
Pydantic contracts for MendSeed — A Public Grievance Observatory.
MendSeed captures and structures real-world problems faced by people across diverse sectors
(civic infrastructure, agriculture, healthcare, education, logistics, governance, etc.)
and packages them for builders, researchers, and hackathon teams seeking genuine problems to solve.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# Sector classifications for real-world grievances
GrievanceSector = Literal[
    "Civic & Public Infrastructure",
    "Agriculture & Rural Development",
    "Healthcare & Sanitation",
    "Education & Skilling",
    "Environment & Waste Management",
    "Transportation & Mobility",
    "Public Services & Local Governance",
    "Livelihoods & Daily Life",
    "Other Grievance"
]


class GrievanceAnalysis(BaseModel):
    """
    Structured extraction of a raw citizen grievance or real-world problem.
    Distills informal, emotional, or fragmented community input into clear problem parameters.
    """
    category: GrievanceSector = Field(
        description="Sector or domain of the real-world grievance"
    )
    normalized_title: str = Field(
        description="Concise, neutral title summarizing the ground reality problem"
    )
    summary: str = Field(
        description="Clear 1-2 sentence distillation of the core problem reported"
    )
    affected_sector: str = Field(
        description="Specific sector, industry, or public domain affected"
    )
    location_or_demographic: Optional[str] = Field(
        default=None,
        description="Geographic region, community group, or demographic facing the problem"
    )
    observed_friction: Optional[str] = Field(
        default=None,
        description="The specific breakdown, shortage, delay, or systemic friction observed"
    )
    current_workarounds: Optional[str] = Field(
        default=None,
        description="Ad-hoc or painful workarounds people currently use to cope"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Critical ground details missing from the report needed to build a solution"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence score"
    )


class GrievanceComparison(BaseModel):
    """
    Compares two community problem reports to identify shared systemic causes
    or distinct regional/domain variations.
    """
    relationship: Literal[
        "Same underlying problem",
        "Related problem in same domain",
        "Distinct problem",
        "Insufficient information"
    ] = Field(
        description="Classified relationship between the new grievance and an existing problem"
    )
    evidence: List[str] = Field(
        description="Shared real-world symptoms, bottlenecks, or failure modes observed"
    )
    matching_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic similarity or compatibility score between grievances"
    )
    meaningful_differences: List[str] = Field(
        default_factory=list,
        description="Differences in local context, demographic, root cause, or severity"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Proposed curation action (e.g., 'cluster_under_canonical', 'keep_distinct')"
    )


class CommunitySummary(BaseModel):
    """
    Synthesized community voice across multiple reports in a sector or region.
    Reveals common patterns for researchers and policy makers.
    """
    main_problems: List[str] = Field(
        description="Primary unresolved pain points reported across the group"
    )
    recurring_themes: List[str] = Field(
        description="Common contexts, conditions, or demographics repeatedly impacted"
    )
    disagreements: List[str] = Field(
        default_factory=list,
        description="Divergent community perspectives, conflicting observations, or disputed causes"
    )
    evidence_backed_feedback: List[str] = Field(
        description="Ground-truth quotes, reported data points, and direct evidence"
    )


class BuilderBrief(BaseModel):
    """
    Actionable problem brief for builders, hackathon teams, and project creators
    looking for real, validated problems to solve.
    """
    problem_statement: str = Field(
        description="Precise articulation of the real-world friction to be solved"
    )
    target_beneficiaries: str = Field(
        description="Who benefits when this problem is solved (e.g., smallholder farmers, bus commuters)"
    )
    current_ground_reality: str = Field(
        description="What currently happens on the ground and why existing options fall short"
    )
    impact: str = Field(
        description="Economic, health, time, or social toll caused by this problem"
    )
    solution_opportunities: List[str] = Field(
        default_factory=list,
        description="Actionable directions, project ideas, or hackathon concepts to address this problem"
    )
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="Key open field questions builders should investigate before designing a solution"
    )


class CanonicalClusterProposal(BaseModel):
    """
    Proposed grouping of related grievances under a canonical problem umbrella.
    Ensures human review is always required before aggregating community records.
    """
    candidate_canonical_problem_id: str = Field(
        description="ID of the overarching canonical problem statement"
    )
    proposed_grievance_links: List[str] = Field(
        default_factory=list,
        description="List of individual citizen grievance IDs to cluster under this problem"
    )
    supporting_evidence: str = Field(
        description="Shared root causes, systemic connections, and field rationale"
    )
    uncertainty_level: Literal["Low", "Medium", "High"] = Field(
        description="Estimated uncertainty in the clustering proposal"
    )
    human_review_required: bool = Field(
        default=True,
        description="Human moderation is strictly mandatory before grouping public records"
    )


# Backward-compatibility aliases so existing code never breaks
IssueAnalysis = GrievanceAnalysis
IssueComparison = GrievanceComparison
DeveloperBrief = BuilderBrief
MergeProposal = CanonicalClusterProposal
