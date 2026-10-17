# MendSeed AI & QuietFlay v0.1 Architecture Rules

## Platform Mission:
MendSeed is a **public grievance observatory**. It captures, structures, and ranks real-world problems across diverse sectors (agriculture, civic infrastructure, healthcare, education, sanitation, mobility, governance) and turns them into **actionable problem briefs for builders, hackathons, and research projects**.

## Critical Constraints for Antigravity IDE:
- **Zero Heavy ML in Core App**: NEVER import `torch`, `transformers`, `peft`, or `bitsandbytes` in the main MendSeed repository or client-facing application layers.
- **SDK Import Boundary**: Always import grievance schemas (`GrievanceAnalysis`, `GrievanceComparison`, `CommunitySummary`, `BuilderBrief`, `CanonicalClusterProposal`), `MendSeedAIClient`, and `QuietFlayEngine` strictly from the `mendseed_ai` SDK package.
- **Asynchronous Outbox Pattern**: All remote AI inference requests must execute asynchronously inside the Transactional Outbox worker (`MODERATION_PENDING` / background worker queue), never inside synchronous database transactions or API request-response locks.
- **QuietFlay Streaming Recurrence**: For real-time event scoring, call `quietflay.update_streaming_state` or `quietflay.evaluate_branching_mass` locally in Python using standard library `math` without network overhead.
- **Human Governance**: All problem merges, canonical cluster links, and moderation actions proposed by the model must require explicit human review (`human_review_required=True`).
