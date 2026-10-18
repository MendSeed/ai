"""
Synthetic Seed Dataset Generator for MendSeed AI.

Generates initial synthetic training, validation, and test examples for development.
NOTE: These examples are machine-generated synthetic seeds for structural testing
and pipeline validation. They must NEVER be described as human-verified or gold-standard.

Outputs:
- training/mendseed_training_data.jsonl (Legacy single-file dataset)
- training/data/train.jsonl (Training partition)
- training/data/val.jsonl (Validation partition)
- training/data/test.jsonl (Locked test partition)
"""

import json
import os
import sys

# Ensure sdk and training modules can be imported from any working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
SDK_DIR = os.path.join(PARENT_DIR, "sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from schema import TrainingExample


def get_seed_examples() -> list[dict]:
    """
    Returns initial synthetic seed examples with explicit cluster isolation
    and task assignments.
    """
    return [
        # =====================================================================
        # Task 1: GrievanceAnalysis (Issue extraction)
        # =====================================================================
        {
            "id": "SEED-ANL-001",
            "task": "grievance_analysis",
            "cluster_id": "CLUST-AGRI-KOLAR-TOMATO",
            "split": "train",
            "category": "Agriculture & Rural Development",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. "
                "Extract structured data from raw citizen reports across diverse sectors. "
                "Return ONLY valid JSON matching the GrievanceAnalysis schema: "
                "category, normalized_title, summary, affected_sector, location_or_demographic, "
                "observed_friction, current_workarounds, missing_information, confidence."
            ),
            "input": (
                "In our block (Kolar district), tomato farmers lose up to 40% of their harvest every June "
                "because there is no solar cold room or refrigerated transit to the wholesale mandi 60km away. "
                "Middlemen exploit this by buying crates at distress prices (Rs 3/kg)."
            ),
            "output": json.dumps({
                "category": "Agriculture & Rural Development",
                "normalized_title": "Lack of decentralized cold storage causes tomato harvest distress sales",
                "summary": "Farmers in Kolar lose up to 40% of tomato harvests due to absent localized cold storage and cold-chain transit to wholesale markets.",
                "affected_sector": "Post-Harvest Agriculture & Cold-Chain Logistics",
                "location_or_demographic": "Kolar district tomato farming community",
                "observed_friction": "Absence of localized or solar-powered cold storage leading to rapid perishability within 48 hours.",
                "current_workarounds": "Selling at distress prices (Rs 3/kg) to predatory intermediaries.",
                "missing_information": ["Daily produce volume during peak harvest", "Grid power reliability in block"],
                "confidence": 0.95
            })
        },
        {
            "id": "SEED-ANL-002",
            "task": "grievance_analysis",
            "cluster_id": "CLUST-CIVIC-WARD14-WATER",
            "split": "train",
            "category": "Civic & Public Infrastructure",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. "
                "Extract structured data from raw citizen reports across diverse sectors. "
                "Return ONLY valid JSON matching the GrievanceAnalysis schema: "
                "category, normalized_title, summary, affected_sector, location_or_demographic, "
                "observed_friction, current_workarounds, missing_information, confidence."
            ),
            "input": (
                "Our municipal tap water in Ward 14 turns brownish and smells like sewage every Tuesday morning. "
                "Over 80 households report children getting stomach infections. Ward office says pipes are old."
            ),
            "output": json.dumps({
                "category": "Civic & Public Infrastructure",
                "normalized_title": "Sewage infiltration into municipal drinking water pipeline",
                "summary": "Weekly sewage contamination in municipal drinking supply causing recurring gastrointestinal infections in 80+ families in Ward 14.",
                "affected_sector": "Urban Water Supply & Public Sanitation",
                "location_or_demographic": "Ward 14 residents (approx. 80 households)",
                "observed_friction": "Negative pressure in aging water lines drawing in adjacent drainage leakage during non-supply hours.",
                "current_workarounds": "Boiling contaminated water or purchasing private RO tanker water.",
                "missing_information": ["Specific street intersections where line crossings occur", "Water sample lab test report"],
                "confidence": 0.94
            })
        },
        {
            "id": "SEED-ANL-003",
            "task": "grievance_analysis",
            "cluster_id": "CLUST-HLTH-DHANKUTA-SNAKE",
            "split": "val",
            "category": "Healthcare & Sanitation",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. "
                "Extract structured data from raw citizen reports across diverse sectors. "
                "Return ONLY valid JSON matching the GrievanceAnalysis schema: "
                "category, normalized_title, summary, affected_sector, location_or_demographic, "
                "observed_friction, current_workarounds, missing_information, confidence."
            ),
            "input": (
                "Primary health sub-centre at Dhankuta has no antivenom vials in stock during monsoon. "
                "A farmer bitten by a viper had to travel 3 hours on dirt roads to district hospital and died on the way."
            ),
            "output": json.dumps({
                "category": "Healthcare & Sanitation",
                "normalized_title": "Primary health centre lacks anti-snake venom and emergency cold storage",
                "summary": "Fatal delay in snakebite envenomation management due to stock-outs of antivenom vials and lack of cold storage at peripheral health centres.",
                "affected_sector": "Rural Emergency Healthcare & Medical Supply Chain",
                "location_or_demographic": "Dhankuta sub-centre catchment area (agricultural laborers)",
                "observed_friction": "Stock-out of temperature-sensitive antivenom at primary point of care during peak monsoon bite season.",
                "current_workarounds": "Hazardous 3-hour emergency transit over unpaved roads to distant district facility.",
                "missing_information": ["Primary health centre electrification status", "Frequency of snakebite incidents per season"],
                "confidence": 0.96
            })
        },
        {
            "id": "SEED-ANL-004",
            "task": "grievance_analysis",
            "cluster_id": "CLUST-EDU-ZILLA-BUS",
            "split": "test",
            "category": "Education & Skilling",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. "
                "Extract structured data from raw citizen reports across diverse sectors. "
                "Return ONLY valid JSON matching the GrievanceAnalysis schema: "
                "category, normalized_title, summary, affected_sector, location_or_demographic, "
                "observed_friction, current_workarounds, missing_information, confidence."
            ),
            "input": (
                "Girls high school in Zilla Parishad has only one functional toilet for 450 students. "
                "During menstruation, attendance drops by 35% because there is no running water or disposal bin."
            ),
            "output": json.dumps({
                "category": "Education & Skilling",
                "normalized_title": "Inadequate menstrual hygiene infrastructure causing girl student absenteeism",
                "summary": "Severe lack of sanitation facilities (1 toilet per 450 students) without running water or disposal causing high dropout and absenteeism.",
                "affected_sector": "School Infrastructure & Adolescent Health",
                "location_or_demographic": "Zilla Parishad Girls High School students (450 students)",
                "observed_friction": "Single unmaintained latrine with zero water connection serving entire school population.",
                "current_workarounds": "Staying home for 4-5 days every month during menstrual cycle.",
                "missing_information": ["Local panchayat fund allocation status", "Borewell water availability on campus"],
                "confidence": 0.95
            })
        },
        {
            "id": "SEED-ANL-005",
            "task": "grievance_analysis",
            "cluster_id": "CLUST-GOV-RATION-BIOMETRIC",
            "split": "val",
            "category": "Public Services & Local Governance",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. "
                "Extract structured data from raw citizen reports across diverse sectors. "
                "Return ONLY valid JSON matching the GrievanceAnalysis schema: "
                "category, normalized_title, summary, affected_sector, location_or_demographic, "
                "observed_friction, current_workarounds, missing_information, confidence."
            ),
            "input": (
                "Elderly pensioners and daily wage laborers in Block 5 fair price ration shop fail biometric fingerprint authentication "
                "repeatedly due to worn-out skin ridges. Ration dealer turns them away without food grains for 3 consecutive months."
            ),
            "output": json.dumps({
                "category": "Public Services & Local Governance",
                "normalized_title": "Biometric fingerprint authentication failure denying ration grains to elderly citizens",
                "summary": "Recurring biometric point-of-sale authentication rejections at fair price shops excluding elderly beneficiaries from statutory food security rations.",
                "affected_sector": "Public Food Distribution & Welfare Technology",
                "location_or_demographic": "Block 5 elderly pensioners and manual laborers (approx. 65 beneficiaries)",
                "observed_friction": "Rigid point-of-sale biometric scanner failure without accessible offline OTP or nominee proxy override.",
                "current_workarounds": "Borrowing grains from neighbors or making repeated 6km visits to the taluk supply office.",
                "missing_information": ["Fair price shop dealer license number", "Availability of iris scan alternative in district"],
                "confidence": 0.96
            })
        },

        # =====================================================================

        # Task 2: GrievanceComparison (Pairwise root cause comparison)
        # =====================================================================
        {
            "id": "SEED-CMP-001",
            "task": "grievance_comparison",
            "cluster_id": "CLUST-TRANS-SHIFT-WORKERS",
            "split": "train",
            "category": "Transportation & Mobility",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI. Compare two community problem reports to identify shared systemic causes. "
                "Return ONLY valid JSON matching GrievanceComparison schema: "
                "relationship ('Same underlying problem' | 'Related problem in same domain' | 'Distinct problem' | 'Insufficient information'), "
                "evidence, matching_score, meaningful_differences, recommended_action."
            ),
            "input": (
                "Grievance A: Rural feeder buses from Taluka headquarter stop running at 6 PM, forcing women garment factory workers on evening shifts to walk 8 km in the dark.\n"
                "Grievance B: Public transport schedule ignores industrial estate shift change at 7:30 PM; female textile workers face severe safety hazards commuting home."
            ),
            "output": json.dumps({
                "relationship": "Same underlying problem",
                "evidence": [
                    "Both reports identify public transport schedules unaligned with women industrial worker shift timings.",
                    "Both highlight extreme vulnerability and lack of last-mile mobility after dark."
                ],
                "matching_score": 0.91,
                "meaningful_differences": [
                    "Grievance A specifies 6 PM cutoff and 8 km walking distance; Grievance B highlights the specific 7:30 PM factory shift change."
                ],
                "recommended_action": "cluster_under_canonical"
            })
        },
        {
            "id": "SEED-CMP-002",
            "task": "grievance_comparison",
            "cluster_id": "CLUST-CROSS-WATER-ROADS",
            "split": "test",
            "category": "Civic & Public Infrastructure",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI. Compare two community problem reports to identify shared systemic causes. "
                "Return ONLY valid JSON matching GrievanceComparison schema: "
                "relationship ('Same underlying problem' | 'Related problem in same domain' | 'Distinct problem' | 'Insufficient information'), "
                "evidence, matching_score, meaningful_differences, recommended_action."
            ),
            "input": (
                "Grievance A: High fluoride content in municipal tubewell causing fluorosis in children.\n"
                "Grievance B: Unpaved rural link road washed away during rains cutting off school buses."
            ),
            "output": json.dumps({
                "relationship": "Distinct problem",
                "evidence": [
                    "Grievance A concerns groundwater chemical toxicity and public health.",
                    "Grievance B concerns physical road surface erosion and transportation access."
                ],
                "matching_score": 0.12,
                "meaningful_differences": [
                    "Completely different physical domains, root failure modes, and required technical interventions."
                ],
                "recommended_action": "keep_distinct"
            })
        },

        # =====================================================================
        # Task 3: BuilderBrief (Hackathon & builder challenge)
        # =====================================================================
        {
            "id": "SEED-BRF-001",
            "task": "builder_brief",
            "cluster_id": "CLUST-LIV-DAILY-CREDIT",
            "split": "train",
            "category": "Livelihoods & Daily Life",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI. Generate an actionable BuilderBrief for hackathon teams, researchers, and project creators "
                "looking for real-world problems to solve. Return ONLY valid JSON matching BuilderBrief schema: "
                "problem_statement, target_beneficiaries, current_ground_reality, impact, solution_opportunities, unresolved_questions."
            ),
            "input": (
                "Problem: Street vendors in small towns pay 10% daily interest (call money) to informal money lenders "
                "for working capital of Rs 1,000 - 3,000 every single morning to buy vegetables."
            ),
            "output": json.dumps({
                "problem_statement": "Daily micro-working capital deficit forcing street vendors into predatory informal credit cycles (10% daily interest rates).",
                "target_beneficiaries": "Urban and semi-urban daily micro-vendors, roadside vegetable and fruit sellers.",
                "current_ground_reality": "Formal microfinance and banks require documentation and multi-day disbursement cycles unsuited for morning wholesale purchase needs (5 AM cash requirement repaid by 8 PM).",
                "impact": "Vendors lose 20-30% of daily livelihood profit to loan sharks, preventing wealth accumulation.",
                "solution_opportunities": [
                    "Intraday UPI-based automated micro-credit pool with instant sunset repayment at market close.",
                    "Community rotating savings and credit association (ROSCA) smartphone coordination tool.",
                    "Cooperative bulk purchasing system to bypass individual morning cash constraints."
                ],
                "unresolved_questions": [
                    "What percentage of targeted vendors possess active UPI-linked smartphones?",
                    "What default mitigation mechanisms work within vendor community trust networks?"
                ]
            })
        },

        # =====================================================================
        # Task 4: CommunitySummary (Synthesis across reports)
        # =====================================================================
        {
            "id": "SEED-SUM-001",
            "task": "community_summary",
            "cluster_id": "CLUST-CIVIC-NH24-GRAVEL",
            "split": "train",
            "category": "Civic & Public Infrastructure",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI. Synthesize recurring community grievances across multiple reports. "
                "Return ONLY valid JSON matching CommunitySummary schema: "
                "main_problems, recurring_themes, disagreements, evidence_backed_feedback."
            ),
            "input": (
                "Public Reports:\n"
                "1. Potholes on NH-24 service road caused 4 two-wheeler skid accidents after light rain.\n"
                "2. Open drainage trench dug by telecom contractors left unfilled for 3 weeks near school entrance.\n"
                "3. Road repair gravel dumped loosely without asphalt roller, causing bike skids."
            ),
            "output": json.dumps({
                "main_problems": [
                    "Uncoordinated utility trenching and uncompacted aggregate gravel creating critical physical hazards on arterial roads."
                ],
                "recurring_themes": [
                    "Lack of inter-agency coordination between municipal road department and private utility contractors.",
                    "Vulnerability of two-wheeler riders and schoolchildren."
                ],
                "disagreements": [],
                "evidence_backed_feedback": [
                    "4 verified two-wheeler skid accidents post-rain on NH-24 service lane",
                    "3-week unbarricaded trench outside school gate"
                ]
            })
        },

        # =====================================================================
        # Task 5: CanonicalClusterProposal (Cautious problem clustering)
        # =====================================================================
        {
            "id": "SEED-MRG-001",
            "task": "canonical_cluster_proposal",
            "cluster_id": "CLUST-ENV-GROUNDWATER",
            "split": "train",
            "category": "Environment & Waste Management",
            "origin": "synthetic_seed",
            "quality_verified": False,
            "system": (
                "You are MendSeed AI. Evaluate whether incoming citizen grievances cluster under a canonical real-world problem statement. "
                "Return ONLY valid JSON matching CanonicalClusterProposal schema: "
                "candidate_canonical_problem_id, proposed_grievance_links, supporting_evidence, uncertainty_level, human_review_required."
            ),
            "input": (
                "Incoming Grievance: Tubewell in Ward 8 is pumping high salinity water, children developing kidney gravel.\n"
                "Canonical Issues: [{'id': 'PROB-GW-204', 'title': 'Groundwater table depletion leading to dangerous fluoride and TDS concentration in sub-district aquifers'}]"
            ),
            "output": json.dumps({
                "candidate_canonical_problem_id": "PROB-GW-204",
                "proposed_grievance_links": ["PROB-GW-204"],
                "supporting_evidence": "Groundwater mineralization and kidney health complications directly stem from sub-aquifer depletion captured in PROB-GW-204.",
                "uncertainty_level": "Low",
                "human_review_required": True
            })
        }
    ]


def generate_datasets(repo_root: str = PARENT_DIR):
    """
    Generates dataset files resolving paths from the repository root.
    """
    examples = get_seed_examples()

    # Validate each example against TrainingExample schema
    valid_examples = []
    for ex in examples:
        model = TrainingExample.model_validate(ex)
        is_valid, err = model.validate_target_contract()
        if not is_valid:
            raise ValueError(f"Seed example {model.id} failed contract validation: {err}")
        valid_examples.append(model)

    # 1. Output main dataset in training directory
    legacy_path = os.path.join(repo_root, "training", "mendseed_training_data.jsonl")
    os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
    with open(legacy_path, "w", encoding="utf-8") as f:
        for ex in valid_examples:
            f.write(ex.model_dump_json() + "\n")
    print(f"Written legacy dataset: {legacy_path} ({len(valid_examples)} synthetic seed examples)")

    # 2. Output partitioned splits in training/data/
    data_dir = os.path.join(repo_root, "training", "data")
    os.makedirs(data_dir, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for ex in valid_examples:
        s = ex.split if ex.split in splits else "train"
        splits[s].append(ex)

    for split_name, split_exs in splits.items():
        split_path = os.path.join(data_dir, f"{split_name}.jsonl")
        with open(split_path, "w", encoding="utf-8") as f:
            for ex in split_exs:
                f.write(ex.model_dump_json() + "\n")
        print(f"Written {split_name} split: {split_path} ({len(split_exs)} examples)")


if __name__ == "__main__":
    generate_datasets()
