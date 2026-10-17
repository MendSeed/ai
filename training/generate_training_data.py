"""
Synthetic Training Dataset Generator for MendSeed AI.
Generates `mendseed_training_data.jsonl` covering all 5 Pydantic contracts:
1. IssueAnalysis
2. IssueComparison
3. CommunitySummary
4. DeveloperBrief
5. MergeProposal
"""

import json
import os
import sys

# Ensure sdk contracts can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)


def generate_dataset(output_path: str = "mendseed_training_data.jsonl"):
    examples = [
        # 1. IssueAnalysis examples
        {
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant. "
                "Return ONLY valid JSON matching the IssueAnalysis schema: "
                "category ('Bug' | 'Feature Request' | 'Inquiry'), normalized_title, summary, "
                "affected_area, reproducible_details, missing_information, confidence."
            ),
            "input": "My map freezes when location services are set to high accuracy. Samsung S21 with Android 13.",
            "output": json.dumps({
                "category": "Bug",
                "normalized_title": "Map freezes on high accuracy GPS",
                "summary": "Application map interface freezes when location services are switched to high accuracy on Samsung S21.",
                "affected_area": "Map / Geolocation",
                "reproducible_details": "Samsung S21 running Android 13 with high accuracy location enabled.",
                "missing_information": [],
                "confidence": 0.95
            })
        },
        {
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant. "
                "Return ONLY valid JSON matching the IssueAnalysis schema: "
                "category ('Bug' | 'Feature Request' | 'Inquiry'), normalized_title, summary, "
                "affected_area, reproducible_details, missing_information, confidence."
            ),
            "input": "Can we get a dark mode option for the dashboard? White background hurts my eyes at night.",
            "output": json.dumps({
                "category": "Feature Request",
                "normalized_title": "Add dark mode theme support to dashboard",
                "summary": "User requests a dark theme for the community dashboard to reduce eye strain in low-light environments.",
                "affected_area": "UI / Dashboard",
                "reproducible_details": None,
                "missing_information": [],
                "confidence": 0.98
            })
        },
        {
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant. "
                "Return ONLY valid JSON matching the IssueAnalysis schema: "
                "category ('Bug' | 'Feature Request' | 'Inquiry'), normalized_title, summary, "
                "affected_area, reproducible_details, missing_information, confidence."
            ),
            "input": "How do I share heirloom tomato seeds with growers in my neighborhood zip code?",
            "output": json.dumps({
                "category": "Inquiry",
                "normalized_title": "How to initiate local seed sharing by zip code",
                "summary": "User seeks guidance on how to exchange heirloom seeds with nearby growers based on postal location.",
                "affected_area": "Seed Exchange / Discovery",
                "reproducible_details": None,
                "missing_information": [],
                "confidence": 0.94
            })
        },

        # 2. IssueComparison examples
        {
            "system": (
                "You are MendSeed AI. Compare two community issue reports. Return ONLY valid JSON matching "
                "IssueComparison schema: relationship ('Same underlying issue' | 'Related issue' | "
                "'Distinct issue' | 'Insufficient information'), evidence, matching_score, "
                "meaningful_differences, recommended_action."
            ),
            "input": (
                "Report A: Map view locks up when zooming in on city center.\n"
                "Report B: Zooming in past level 16 causes the entire application interface to freeze."
            ),
            "output": json.dumps({
                "relationship": "Same underlying issue",
                "evidence": ["Both reports indicate UI freeze specifically triggered by map zooming operations."],
                "matching_score": 0.89,
                "meaningful_differences": ["Report B provides a specific zoom threshold (level 16)."],
                "recommended_action": "link_duplicate"
            })
        },

        # 3. CommunitySummary examples
        {
            "system": (
                "You are MendSeed AI. Synthesize recurring problems across community reports. "
                "Return ONLY valid JSON matching CommunitySummary schema: main_problems, "
                "recurring_themes, disagreements, evidence_backed_feedback."
            ),
            "input": (
                "Reports:\n"
                "1. Soil test upload rejects PDF documents larger than 2MB.\n"
                "2. Uploading photo of my garden returns timeout error on 4G connection.\n"
                "3. Image upload fails silently without error message."
            ),
            "output": json.dumps({
                "main_problems": ["File and photo uploads fail or time out under low bandwidth or moderate file sizes."],
                "recurring_themes": ["Asset ingestion pipeline fragility", "Missing client-side error notifications"],
                "disagreements": [],
                "evidence_backed_feedback": [
                    "2MB PDF size limit exceeded",
                    "Timeout on 4G cellular connection",
                    "Silent failure on image upload"
                ]
            })
        },

        # 4. DeveloperBrief examples
        {
            "system": (
                "You are MendSeed AI. Generate a concise DeveloperBrief for software maintainers. "
                "Return ONLY valid JSON matching DeveloperBrief schema: problem_statement, "
                "expected_behavior, actual_behavior, impact, reproduction_steps, unresolved_questions."
            ),
            "input": (
                "Canonical Title: Seed swap transaction fails with out of stock error despite counter showing 5\n"
                "Evidence: Users clicked 'Request Seed', counter showed 5 available, error prompt 'Item no longer in inventory'."
            ),
            "output": json.dumps({
                "problem_statement": "Race condition in inventory counter decrements during concurrent swap requests.",
                "expected_behavior": "Available stock displays accurate real-time inventory or gracefully locks during checkout.",
                "actual_behavior": "Counter shows positive quantity but checkout throws inventory exhaustion error.",
                "impact": "Blocks legitimate exchanges and causes user distrust in platform inventory accuracy.",
                "reproduction_steps": [
                    "Navigate to listing with low remaining inventory",
                    "Simultaneously initiate two requests from different accounts",
                    "Observe out-of-stock failure on one request"
                ],
                "unresolved_questions": [
                    "Are inventory deductions wrapped in an atomic SQL row lock (SELECT FOR UPDATE)?"
                ]
            })
        },

        # 5. MergeProposal examples
        {
            "system": (
                "You are MendSeed AI. Evaluate merge proposals. "
                "Return ONLY valid JSON matching MergeProposal schema: candidate_canonical_issue_id, "
                "proposed_duplicate_links, supporting_evidence, uncertainty_level, human_review_required."
            ),
            "input": (
                "New Report: Can't reset password because verification email never arrives in inbox or spam.\n"
                "Candidates: [{'id': 'CANON-104', 'title': 'Transactional signup and reset emails delayed by 45 minutes'}]"
            ),
            "output": json.dumps({
                "candidate_canonical_issue_id": "CANON-104",
                "proposed_duplicate_links": ["CANON-104"],
                "supporting_evidence": "Both reports stem from transactional email delivery delays/failures in the authentication flow.",
                "uncertainty_level": "Low",
                "human_review_required": True
            })
        }
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Dataset successfully created with {len(examples)} examples: {output_path}")


if __name__ == "__main__":
    generate_dataset()
