"""
MendSeed AI SDK — Zero-ML Lightweight Client & QuietFlay Engine.
"""

from .contracts import (
    IssueAnalysis,
    IssueComparison,
    CommunitySummary,
    DeveloperBrief,
    MergeProposal,
)
from .quietflay import QuietFlayEngine
from .client import MendSeedAIClient

__version__ = "0.1.0"

__all__ = [
    "IssueAnalysis",
    "IssueComparison",
    "CommunitySummary",
    "DeveloperBrief",
    "MergeProposal",
    "QuietFlayEngine",
    "MendSeedAIClient",
]
