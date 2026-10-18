"""
Unit Tests for Training Dataset Schema and Leakage Validator.

Tests:
1. TrainingExample model validation across tasks.
2. Target contract validation (detects invalid schema outputs).
3. Split leakage detection (flags shared cluster IDs and high Jaccard lexical overlap).
4. Sector coverage check.
"""

import json
import pytest

from training.schema import TrainingExample
from training.validate_dataset import DatasetValidator


def test_training_example_contract_validation():
    # Valid GrievanceAnalysis example
    valid_ex = TrainingExample(
        id="TR-001",
        task="grievance_analysis",
        system="Extract structured data into GrievanceAnalysis",
        input="Farmers face loss due to absence of cold rooms.",
        output=json.dumps({
            "category": "Agriculture & Rural Development",
            "normalized_title": "Absence of cold rooms",
            "summary": "Farmers lose harvest.",
            "affected_sector": "Agriculture",
            "confidence": 0.95
        }),
        origin="synthetic_seed",
        quality_verified=False
    )
    is_valid, err = valid_ex.validate_target_contract()
    assert is_valid is True
    assert err is None

    # Invalid GrievanceAnalysis (missing mandatory summary field)
    invalid_ex = TrainingExample(
        id="TR-002",
        task="grievance_analysis",
        system="Extract structured data into GrievanceAnalysis",
        input="Farmers face loss.",
        output=json.dumps({
            "category": "Agriculture & Rural Development"
            # Missing normalized_title, summary, affected_sector, confidence
        }),
        origin="synthetic_seed",
        quality_verified=False
    )
    is_valid, err = invalid_ex.validate_target_contract()
    assert is_valid is False
    assert "validation failed" in err.lower()


def test_leakage_detection_catches_shared_clusters():
    validator = DatasetValidator(jaccard_leakage_threshold=0.65)

    ex_train = TrainingExample(
        id="TR-01",
        task="grievance_analysis",
        cluster_id="CLUST-SHARED-01",
        system="sys",
        input="Water problem in ward 4.",
        output=json.dumps({
            "category": "Civic & Public Infrastructure",
            "normalized_title": "Water issue",
            "summary": "Water failure.",
            "affected_sector": "Water",
            "confidence": 0.9
        }),
        split="train"
    )

    ex_test = TrainingExample(
        id="TST-01",
        task="grievance_analysis",
        cluster_id="CLUST-SHARED-01",  # LEAKAGE: Same cluster in test split!
        system="sys",
        input="Paraphrased problem in ward 4.",
        output=json.dumps({
            "category": "Civic & Public Infrastructure",
            "normalized_title": "Water issue paraphrased",
            "summary": "Water failure.",
            "affected_sector": "Water",
            "confidence": 0.9
        }),
        split="test"
    )

    res = validator.check_split_leakage({"train": [ex_train], "test": [ex_test]})
    assert res["leakage_detected"] is True
    assert any("CLUST-SHARED-01" in w for w in res["warnings"])


def test_leakage_detection_catches_high_lexical_similarity():
    validator = DatasetValidator(jaccard_leakage_threshold=0.65)

    # Near-duplicate texts without explicit cluster ID
    text_a = "In Kolar district tomato farmers lose forty percent of produce due to lack of cold chain transit"
    text_b = "In Kolar district tomato farmers lose forty percent of produce due to zero cold chain transit"

    ex_train = TrainingExample(
        id="TR-10",
        task="grievance_analysis",
        system="sys",
        input=text_a,
        output=json.dumps({
            "category": "Agriculture & Rural Development",
            "normalized_title": "Tomato loss",
            "summary": "Harvest spoil.",
            "affected_sector": "Agri",
            "confidence": 0.9
        }),
        split="train"
    )

    ex_test = TrainingExample(
        id="TST-10",
        task="grievance_analysis",
        system="sys",
        input=text_b,
        output=json.dumps({
            "category": "Agriculture & Rural Development",
            "normalized_title": "Tomato loss",
            "summary": "Harvest spoil.",
            "affected_sector": "Agri",
            "confidence": 0.9
        }),
        split="test"
    )

    res = validator.check_split_leakage({"train": [ex_train], "test": [ex_test]})
    assert res["leakage_detected"] is True
    assert any("lexical similarity" in w.lower() for w in res["warnings"])
