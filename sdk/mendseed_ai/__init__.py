"""
MendSeed AI SDK — Zero-ML Lightweight Client & QuietFlay Engine.
Public Grievances & Problem Observatory for Builders and Hackathons.
"""

from .contracts import (
    GrievanceAnalysis,
    GrievanceComparison,
    CommunitySummary,
    BuilderBrief,
    CanonicalClusterProposal,
    # Aliases for backward compatibility
    IssueAnalysis,
    IssueComparison,
    DeveloperBrief,
    MergeProposal,
    GrievanceSector,
)
from .quietflay import QuietFlayEngine
from .client import MendSeedAIClient

__version__ = "0.1.0"

__all__ = [
    "GrievanceAnalysis",
    "GrievanceComparison",
    "CommunitySummary",
    "BuilderBrief",
    "CanonicalClusterProposal",
    "IssueAnalysis",
    "IssueComparison",
    "DeveloperBrief",
    "MergeProposal",
    "GrievanceSector",
    "QuietFlayEngine",
    "MendSeedAIClient",
]
