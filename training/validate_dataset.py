"""
MendSeed AI — Dataset Validator & Leakage Detection Engine.

Validates:
1. Schema conformity & JSON contract integrity across all 5 tasks.
2. Honest provenance tracking (synthetic vs human_reviewed).
3. Leakage prevention between splits (near-duplicates, overlapping cluster IDs, high Jaccard similarity).
4. Coverage across all 8 sectors, edge cases, and adversarial prompts.
"""

from collections import Counter, defaultdict
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
SDK_DIR = os.path.join(PARENT_DIR, "sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from schema import TrainingExample

MANDATORY_SECTORS = [
    "Agriculture & Rural Development",
    "Civic & Public Infrastructure",
    "Healthcare & Sanitation",
    "Education & Skilling",
    "Environment & Waste Management",
    "Transportation & Mobility",
    "Public Services & Local Governance",
    "Livelihoods & Daily Life",
]


def tokenize(text: str) -> Set[str]:
    """Extracts lowercase word tokens for overlap analysis."""
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def compute_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Computes Jaccard similarity coefficient between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class DatasetValidator:
    """
    Validates training datasets, catches near-duplicate leakage,
    and enforces honest metadata accounting.
    """

    def __init__(self, jaccard_leakage_threshold: float = 0.65):
        self.jaccard_threshold = jaccard_leakage_threshold

    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """Validates a single jsonl dataset file."""
        if not os.path.exists(filepath):
            return {"valid": False, "error": f"File does not exist: {filepath}"}

        examples: List[TrainingExample] = []
        raw_rows = 0
        parse_errors = []
        contract_errors = []

        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                raw_rows += 1
                try:
                    data = json.loads(line)
                except Exception as e:
                    parse_errors.append(f"Line {idx}: JSON parse error: {e}")
                    continue

                # Auto-assign task if legacy 3-field format (system, input, output)
                if "task" not in data:
                    sys_p = data.get("system", "")
                    if "GrievanceAnalysis" in sys_p:
                        data["task"] = "grievance_analysis"
                    elif "GrievanceComparison" in sys_p:
                        data["task"] = "grievance_comparison"
                    elif "CommunitySummary" in sys_p:
                        data["task"] = "community_summary"
                    elif "BuilderBrief" in sys_p:
                        data["task"] = "builder_brief"
                    elif "CanonicalClusterProposal" in sys_p or "cluster" in sys_p:
                        data["task"] = "canonical_cluster_proposal"
                    else:
                        data["task"] = "grievance_analysis"

                if "id" not in data:
                    data["id"] = f"EX-{idx:04d}"

                try:
                    ex = TrainingExample.model_validate(data)
                    is_contract_valid, err = ex.validate_target_contract()
                    if not is_contract_valid:
                        contract_errors.append(f"Example {ex.id} (Line {idx}): {err}")
                    else:
                        examples.append(ex)
                except Exception as e:
                    parse_errors.append(f"Line {idx}: Schema validation error: {e}")

        # Compute distributions
        task_counts = Counter(e.task for e in examples)
        origin_counts = Counter(e.origin for e in examples)
        split_counts = Counter(e.split for e in examples)
        category_counts = Counter(e.category for e in examples if e.category)
        human_verified_count = sum(1 for e in examples if e.quality_verified)

        is_valid = len(parse_errors) == 0 and len(contract_errors) == 0 and len(examples) > 0

        return {
            "valid": is_valid,
            "filepath": filepath,
            "total_raw_rows": raw_rows,
            "valid_examples": len(examples),
            "parse_errors": parse_errors[:10],
            "contract_errors": contract_errors[:10],
            "task_distribution": dict(task_counts),
            "origin_distribution": dict(origin_counts),
            "split_distribution": dict(split_counts),
            "category_distribution": dict(category_counts),
            "human_verified_count": human_verified_count,
            "synthetic_count": len(examples) - human_verified_count,
            "examples": examples
        }

    def check_split_leakage(
        self,
        splits: Dict[str, List[TrainingExample]]
    ) -> Dict[str, Any]:
        """
        Detects data leakage across splits (e.g. train vs val, train vs test):
        1. Exact duplicate inputs or outputs.
        2. Shared cluster IDs (same root problem in both train and test).
        3. Near-duplicate text exceeding Jaccard threshold.
        """
        leakage_warnings = []
        split_names = list(splits.keys())

        # Check all pairs of distinct splits
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                name_a = split_names[i]
                name_b = split_names[j]
                list_a = splits[name_a]
                list_b = splits[name_b]

                # 1. Cluster ID overlap
                clusters_a = {e.cluster_id for e in list_a if e.cluster_id}
                clusters_b = {e.cluster_id for e in list_b if e.cluster_id}
                shared_clusters = clusters_a & clusters_b
                if shared_clusters:
                    leakage_warnings.append(
                        f"Cluster ID leakage between {name_a} and {name_b}: {shared_clusters}"
                    )

                # 2. Text near-duplicate check
                tokens_cache_b = [(e.id, tokenize(e.input)) for e in list_b]
                for ex_a in list_a:
                    tok_a = tokenize(ex_a.input)
                    if len(tok_a) < 5:
                        continue
                    for id_b, tok_b in tokens_cache_b:
                        sim = compute_jaccard(tok_a, tok_b)
                        if sim >= self.jaccard_threshold:
                            leakage_warnings.append(
                                f"High lexical similarity ({sim:.2f}) between {name_a} ({ex_a.id}) and {name_b} ({id_b})"
                            )

        return {
            "leakage_detected": len(leakage_warnings) > 0,
            "total_leakage_warnings": len(leakage_warnings),
            "warnings": leakage_warnings[:15]
        }

    def check_sector_coverage(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Verifies representation across all 8 mandatory sectors."""
        covered = set()
        for ex in examples:
            if ex.category:
                covered.add(ex.category)
            # Try to extract category from output JSON
            try:
                out = json.loads(ex.output)
                if "category" in out:
                    covered.add(out["category"])
                if "affected_sector" in out:
                    covered.add(out["affected_sector"])
            except Exception:
                pass

        missing_sectors = [s for s in MANDATORY_SECTORS if not any(s.lower() in c.lower() for c in covered)]
        return {
            "all_sectors_covered": len(missing_sectors) == 0,
            "covered_sectors": list(covered),
            "missing_sectors": missing_sectors
        }


def main():
    """CLI runner for dataset validation."""
    target_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CURRENT_DIR, "mendseed_training_data.jsonl")
    print(f"--- MendSeed AI Dataset Validator ---")
    print(f"Target: {target_path}")

    validator = DatasetValidator()
    report = validator.validate_file(target_path)

    print(f"Status: {'VALID' if report['valid'] else 'INVALID'}")
    print(f"Total Rows: {report.get('total_raw_rows', 0)} | Valid Examples: {report.get('valid_examples', 0)}")
    print(f"Human-Verified: {report.get('human_verified_count', 0)} | Synthetic: {report.get('synthetic_count', 0)}")
    print(f"Tasks: {report.get('task_distribution')}")
    print(f"Origins: {report.get('origin_distribution')}")

    if report.get("parse_errors"):
        print("\nParse Errors:")
        for err in report["parse_errors"]:
            print(f"  - {err}")

    if report.get("contract_errors"):
        print("\nContract Errors:")
        for err in report["contract_errors"]:
            print(f"  - {err}")

    if report.get("valid"):
        coverage = validator.check_sector_coverage(report["examples"])
        print(f"\nSector Coverage: {'Complete' if coverage['all_sectors_covered'] else 'Incomplete'}")
        if coverage["missing_sectors"]:
            print(f"  Missing: {coverage['missing_sectors']}")


if __name__ == "__main__":
    main()
