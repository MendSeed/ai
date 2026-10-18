"""
Training Data Schema & Specification for MendSeed AI.

Enforces typed schemas, task definitions, and origin provenance
for all training, validation, and evaluation records across the 5 tasks.
"""

import json
import os
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

# Ensure sdk is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
SDK_DIR = os.path.join(PARENT_DIR, "sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from mendseed_ai.contracts import (
    BuilderBrief,
    CanonicalClusterProposal,
    CommunitySummary,
    GrievanceAnalysis,
    GrievanceComparison,
)

TaskType = Literal[
    "grievance_analysis",
    "grievance_comparison",
    "community_summary",
    "builder_brief",
    "canonical_cluster_proposal"
]

DataOrigin = Literal[
    "synthetic_seed",
    "synthetic_augmented",
    "human_reviewed",
    "adversarial_stress",
    "field_report"
]

DataSplit = Literal[
    "train",
    "val",
    "test",
    "unassigned"
]


class TrainingExample(BaseModel):
    """
    Standard format for an individual MendSeed AI training or evaluation record.
    """
    id: str = Field(description="Unique record identifier, e.g. TR-AGRI-001")
    task: TaskType = Field(description="Target MendSeed task")
    system: str = Field(description="System instruction prompt")
    input: str = Field(description="User prompt or input text")
    output: str = Field(description="Ground truth or target output JSON string")
    origin: DataOrigin = Field(
        default="synthetic_seed",
        description="Data provenance: synthetic_seed, human_reviewed, adversarial_stress, etc."
    )
    quality_verified: bool = Field(
        default=False,
        description="Strictly False for synthetic data; True ONLY after human expert review"
    )
    split: DataSplit = Field(default="unassigned", description="Dataset split assignment")
    category: Optional[str] = Field(default=None, description="Domain or sector label")
    cluster_id: Optional[str] = Field(
        default=None,
        description="Underlying problem cluster ID used to prevent train/val/test leakage"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provenance and curation notes")

    def validate_target_contract(self) -> Tuple[bool, Optional[str]]:
        """
        Validates that the output JSON string parses and conforms to the Pydantic contract
        for this example's designated task.
        """
        try:
            parsed = json.loads(self.output)
        except Exception as e:
            return False, f"Output is not valid JSON: {e}"

        try:
            if self.task == "grievance_analysis":
                GrievanceAnalysis.model_validate(parsed)
            elif self.task == "grievance_comparison":
                GrievanceComparison.model_validate(parsed)
            elif self.task == "community_summary":
                CommunitySummary.model_validate(parsed)
            elif self.task == "builder_brief":
                BuilderBrief.model_validate(parsed)
            elif self.task == "canonical_cluster_proposal":
                CanonicalClusterProposal.model_validate(parsed)
            else:
                return False, f"Unknown task: {self.task}"
            return True, None
        except Exception as e:
            return False, f"Contract validation failed for task {self.task}: {e}"
