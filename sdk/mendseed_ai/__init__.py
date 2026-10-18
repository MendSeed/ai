"""
MendSeed AI SDK — Zero-ML Lightweight Client & QuietFray Engine.
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
    PLATFORM_CATEGORY_TO_SECTOR,
    SECTOR_TO_PLATFORM_CATEGORY,
    map_platform_category_to_sector,
    map_sector_to_platform_category,
)
from .quietfray import QuietFrayEngine, QuietFlayEngine
from .client import MendSeedAIClient

__version__ = "0.2.0"

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
    "PLATFORM_CATEGORY_TO_SECTOR",
    "SECTOR_TO_PLATFORM_CATEGORY",
    "map_platform_category_to_sector",
    "map_sector_to_platform_category",
    "QuietFrayEngine",
    "QuietFlayEngine",
    "MendSeedAIClient",
]
