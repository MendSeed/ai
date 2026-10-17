"""
Synthetic Training Dataset Generator for MendSeed AI.
MendSeed is a public grievance observatory that collects real-world problems across sectors
(agriculture, civic infrastructure, healthcare, education, logistics, public services)
and formats them for builders, researchers, and hackathon teams looking for real problems to solve.

Generates `training/mendseed_training_data.jsonl` covering all 5 Pydantic contracts:
1. GrievanceAnalysis (IssueAnalysis)
2. GrievanceComparison (IssueComparison)
3. CommunitySummary
4. BuilderBrief (DeveloperBrief)
5. CanonicalClusterProposal (MergeProposal)
"""

import json
import os
import sys

# Ensure sdk directory is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)


def generate_dataset(output_path: str = "training/mendseed_training_data.jsonl"):
    examples = [
        # -------------------------------------------------------------
        # 1. GrievanceAnalysis: Agriculture / Perishable Harvest Spoilage
        # -------------------------------------------------------------
        {
            "system": (
                "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. "
                "Extract structured data from raw citizen reports across diverse sectors. "
                "Return ONLY valid JSON matching the GrievanceAnalysis schema: "
                "category ('Civic & Public Infrastructure' | 'Agriculture & Rural Development' | "
                "'Healthcare & Sanitation' | 'Education & Skilling' | 'Environment & Waste Management' | "
                "'Transportation & Mobility' | 'Public Services & Local Governance' | 'Livelihoods & Daily Life' | 'Other Grievance'), "
                "normalized_title, summary, affected_sector, location_or_demographic, observed_friction, "
                "current_workarounds, missing_information, confidence."
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
                "confidence": 0.96
            })
        },

        # -------------------------------------------------------------
        # 2. GrievanceAnalysis: Civic Infrastructure / Water Contamination
        # -------------------------------------------------------------
        {
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
                "confidence": 0.95
            })
        },

        # -------------------------------------------------------------
        # 3. GrievanceAnalysis: Healthcare / Remote Emergency Care
        # -------------------------------------------------------------
        {
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
                "confidence": 0.98
            })
        },

        # -------------------------------------------------------------
        # 4. GrievanceComparison: Real-World Public Transportation
        # -------------------------------------------------------------
        {
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
                "meaningful_differences": ["Grievance A specifies 6 PM cutoff and 8 km distance; Grievance B highlights the specific 7:30 PM factory shift change."],
                "recommended_action": "cluster_under_canonical"
            })
        },

        # -------------------------------------------------------------
        # 5. BuilderBrief: Hackathon & Project Problem Definition
        # -------------------------------------------------------------
        {
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

        # -------------------------------------------------------------
        # 6. CommunitySummary: Public Grievance Aggregation
        # -------------------------------------------------------------
        {
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

        # -------------------------------------------------------------
        # 7. CanonicalClusterProposal: Clustering Grievances
        # -------------------------------------------------------------
        {
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

    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"MendSeed Public Grievance Dataset created with {len(examples)} examples: {output_path}")


if __name__ == "__main__":
    generate_dataset()
