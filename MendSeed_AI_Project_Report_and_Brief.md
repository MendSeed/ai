# MendSeed AI — Project Report and Technical Brief

**Document version:** 1.0  
**Date:** 16 October 2026  
**Status:** Prototype planning; base-model smoke test is the immediate next milestone

> **Purpose:** A shared source of truth for what MendSeed AI is meant to do, why it exists, how it should be built, what is already established, and what must happen next.


## 1. Executive summary

MendSeed AI is the AI and issue-intelligence component of MendSeed: an open-source, privacy-first, community-driven platform for collecting real-world problems, connecting people who experience similar problems, and helping developers discover problems worth solving.

The central job is to turn noisy, incomplete, differently worded reports into structured, searchable, comparable, and developer-actionable information. AI should help the community understand patterns and propose links between reports; it must not replace community judgment or silently rewrite the community’s records.

The first release should be a reliable, narrowly scoped assistant—not a general-purpose chatbot and not a model trained from scratch. Start with a pretrained instruction model, a small reviewed dataset, an embedding-based retrieval layer, strict schema validation, and human review for consequential actions. Fine-tuning is justified only after a baseline is measured and representative examples show where the base model fails.


## 2. The problem and the opportunity

People describe the same underlying problem in different ways. One person may report that an editor freezes on a large project; another may say that opening a repository with thousands of files causes the application to hang. A conventional keyword search can miss the relationship. A generic language model may recognize the relationship but invent context, merge different problems, or be inconsistent across requests.

MendSeed needs to combine three kinds of intelligence: structured extraction, semantic retrieval, and cautious comparison. The LLM interprets language; embeddings retrieve potentially related records; explicit rules and human review control what may happen to the source records.

- Less noise: normalize descriptions without erasing the author’s meaning.
- Better discovery: find duplicates and related reports even when their wording differs.
- Better collaboration: summarize recurring pain points and turn them into useful developer briefs.
- Better governance: show evidence and uncertainty, preserve original reports, and leave consequential decisions to people.


## 3. Vision, goals, and non-goals


### Primary goals

- Extract consistent fields from free-text issue reports and return schema-valid JSON.
- Find likely duplicate and related issues using semantic retrieval plus comparison.
- Explain why two reports appear related, including shared symptoms and meaningful differences.
- Generate concise theme summaries and developer-ready briefs grounded in actual reports.
- Flag missing details, sensitive information, spam, abuse, and uncertainty for review.
- Offer a versioned, documented API that the MendSeed application can call reliably.
- Keep source reports and community decisions authoritative; make AI suggestions auditable and reversible.


### Non-goals for the first release

- Training a foundation model from scratch.
- Automatically merging, deleting, hiding, or publishing issue records.
- Claiming that a similarity score proves two reports are duplicates.
- Replacing human moderation, project maintainers, or community voting.
- Using private or personally identifying reports as training data without a clear, consent-based policy.
- Building a general coding agent that executes arbitrary tools or changes repositories.


## 4. What MendSeed AI should do


### A. Issue analysis and normalization

Input: a free-text report and optional safe metadata such as product, version, operating system, or category. Output: a consistent representation that separates what the author said from what is inferred.

Proposed fields: title, summary, category, affected_feature, observed_symptom, trigger_or_steps, expected_behavior, actual_behavior, environment, severity, missing_information, privacy_flags, and confidence or review notes. Unknown facts must stay unknown; they must not be guessed. Severity should be unknown unless the report or an explicit policy supports a classification.


### B. Duplicate and related-issue discovery

The system should first retrieve a small set of candidate reports with embeddings, then compare each candidate against the new report. A comparison should return one of four labels: duplicate, related_but_distinct, distinct, or insufficient_information. It should include a short evidence-based explanation, meaningful differences, confidence, and a recommended review action.

A duplicate means the reports describe substantially the same underlying issue, not merely the same product or feature. “Related” means there is meaningful overlap but the reports should remain separate. The system must make it easy to choose “not a duplicate” and should learn from reviewer decisions only through an explicit, versioned data process.


### C. Community-level theme summaries

Across a selected group of reports, summarize recurring symptoms, affected workflows, contexts, and unresolved questions. Every summary should be traceable to the reports it represents. The model must not turn a small number of reports into a claim about the entire user base.


### D. Developer-ready briefs

Generate a brief with: problem statement, user impact as reported, observed conditions, reproduction details if provided, expected versus actual behavior, related reports, evidence, known unknowns, and questions for maintainers. Separate reported facts from hypotheses. Do not invent fixes, root causes, usage counts, or affected versions.


### E. Safety and privacy assistance

Detect possible personal data, credentials, access tokens, private URLs, abusive content, and likely spam. Treat detection as a review aid, not an infallible verdict. Prefer redaction before model processing when feasible, minimize data sent to inference providers, and avoid recording raw issue text in application logs.


## 5. Proposed architecture

Use a modular pipeline so the language model is one component rather than the entire system. Each stage should have a small, testable contract.

| Layer | Responsibility | Why it exists |

| --- | --- | --- |

| MendSeed application | Collect reports, show suggestions, obtain human decisions, preserve originals. | Keeps community control in the product workflow. |

| API service | FastAPI endpoints, Pydantic request/response validation, authentication and rate limits. | Creates a stable boundary between the app and AI internals. |

| Policy and validation | Input limits, privacy checks, JSON/schema validation, confidence gates, safe fallbacks. | Prevents malformed model output from directly changing records. |

| LLM | Extraction, comparison explanations, summaries, developer briefs. | Handles language understanding and structured generation. |

| Embedding retrieval | Represent reports as vectors and retrieve semantically similar candidates. | Finds candidates efficiently across differently worded reports. |

| Database | PostgreSQL for records and decisions; pgvector for embeddings if appropriate. | Keeps source data, review outcomes, and search data in a consistent system. |

| Evaluation and observability | Golden test sets, metrics, version IDs, latency/error tracking without sensitive payloads. | Makes quality measurable and regressions visible. |

The LLM should not decide on its own what is saved to the canonical database. The API should validate the response, attach model/version metadata, and return a proposal. The application then presents the proposal to a user or moderator. Only an explicit authorized action changes the record.


### Illustrative issue-analysis flow

```json
User report → privacy/input checks → LLM structured extraction → JSON schema validation → embedding generation → candidate retrieval → duplicate/related comparison → review proposal → human decision → audit event
```


### Separation of concerns

- LLM: understand and explain text; do not treat it as the database or source of truth.
- Embeddings: retrieve candidates; do not make final duplicate decisions by themselves.
- Database: hold original reports, canonical records, decisions, and audit events.
- Policy layer: enforce limits, schema contracts, permissions, and approval requirements.
- Human reviewer: approve or reject merges, moderation actions, and publishable summaries.


## 6. Model strategy


### Initial candidate: Qwen3-4B-Instruct-2507

The current candidate for the first language-model experiment is Qwen/Qwen3-4B-Instruct-2507. It is an instruction-tuned 4B-parameter model with an Apache-2.0 license listed on its Hugging Face model card. Re-check the exact model card and license at the pinned revision before distributing weights or a derived adapter. This is a candidate, not a proven final choice.

Why start here: a model in this size range is a reasonable experiment for structured extraction and concise text tasks on a single-GPU notebook. It is not automatically the best model for every task. We must compare its unmodified baseline against a small held-out set before deciding whether fine-tuning is worth the time.


### Fine-tuning approach: supervised fine-tuning with QLoRA

QLoRA loads the base model in 4-bit quantized form and trains small LoRA adapter parameters rather than updating all base weights. This is a memory-saving approach for experiments on limited GPU hardware. Use the current Hugging Face Transformers, PEFT, TRL, and bitsandbytes documentation as the source of truth for APIs because library interfaces change.

The intended stack is Python, PyTorch, Transformers, Datasets, TRL SFTTrainer, PEFT LoRA, and bitsandbytes. A reasonable initial experiment is NF4 4-bit quantization, double quantization, float16 compute on the current T4, rank 16 and alpha 32 LoRA, dropout 0.05, one or two short epochs, micro-batch size 1 with gradient accumulation, and a conservative sequence length. These are starting values for a smoke test, not a guaranteed optimal recipe. Adjust based on actual memory, warnings, loss curves, and validation results.


### Do not train before measuring the baseline

1. Create a small, reviewed evaluation set that is never used for training.
2. Run the base model on those same examples with a fixed prompt and schema.
3. Record parse/schema success, unsupported claims, useful extraction, and failure cases.
4. Train a first adapter on reviewed examples that target observed failures.
5. Re-run exactly the same evaluation and compare per-task results.
6. Keep the adapter only if it produces a clear improvement without unacceptable regressions.

Fine-tuning teaches the model the desired task style and output behavior; it is not the best place to store live issue facts. Fresh community issues should be retrieved from the database at inference time. This avoids repeatedly retraining the model just because the issue collection changes.


## 7. Embeddings and duplicate detection

The embedding model should be evaluated independently from the LLM. A small sentence-embedding model such as BAAI/bge-small-en-v1.5 is a candidate baseline, not a final selection. Use a known-good benchmark made from MendSeed-style issue pairs rather than selecting only from generic leaderboard numbers.

Recommended decision pipeline:

1. Normalize the report and create an embedding.
2. Retrieve top-k similar reports from the vector index, while also applying safe metadata filters when relevant.
3. Compare the new report and each candidate using explicit labels: duplicate, related_but_distinct, distinct, insufficient_information.
4. Return evidence, differences, confidence, and the candidate record IDs.
5. Route low-confidence or ambiguous matches to review; do not silently merge records.
6. Capture the reviewer’s decision as labeled evaluation data, subject to privacy and retention rules.

Evaluate retrieval separately from final comparison. “Did the right candidate appear in the top 5?” is a retrieval question. “Was the candidate correctly classified as duplicate?” is a classification question. Blending these metrics can conceal failures.


## 8. Dataset and labeling plan


### Dataset format

Use versioned JSONL examples for supervised fine-tuning. Each row can contain a messages array with system, user, and assistant turns. The assistant target should be the exact schema-compliant response expected from production. Keep one example focused on one task so failures are easier to diagnose.

```json
{
  "messages": [
    {"role": "system", "content": "Extract issue fields. Do not invent missing facts."},
    {"role": "user", "content": "The editor freezes when opening projects with thousands of files."},
    {"role": "assistant", "content": "{\"category\":\"performance\",\"severity\":\"unknown\",\"missing_information\":[\"project size threshold\"]}"}
  ]
}
```


### First dataset milestone

Target an initial 500–1,000 carefully reviewed examples across the task families, if available. Quality matters more than count; do not pad the set with synthetic or automatically labeled examples and then mistake them for ground truth. Start smaller when necessary, record where examples came from, and expand based on measured gaps.

- Issue extraction: clean reports, incomplete reports, ambiguous reports, multiple issues in one report.
- Comparison pairs: duplicates, related-but-distinct pairs, clearly distinct pairs, and insufficient-information pairs.
- Summaries: examples grounded in a known group of source issues, with citations or source IDs.
- Privacy/moderation: synthetic or consented examples for secrets, personal data, spam, and abuse.
- Edge cases: contradictions, version mismatches, sarcasm, non-English text if in scope, and prompt injection inside user-submitted reports.

Split data by issue group, project/source, or collection period to reduce leakage. Near-duplicate reports must not be spread between train and test in a way that makes evaluation unrealistically easy. Store a manifest with dataset version, provenance, labeler/reviewer status, and permitted uses.


## 9. API contract and product integration

| Endpoint (proposed) | Purpose | Expected behavior |

| --- | --- | --- |

| GET /health | Liveness/readiness basics. | Return service status without leaking secrets or infrastructure details. |

| GET /v1/models | Expose active model and version metadata. | Return model IDs, adapter ID, schema version, and capabilities. |

| POST /v1/issues/analyze | Normalize an issue report. | Return schema-validated fields, unknowns, privacy flags, and review notes. |

| POST /v1/issues/match | Find duplicate/related candidates. | Return candidate IDs, labels, evidence, confidence, and differences. |

| POST /v1/issues/merge-proposal | Prepare a possible merge proposal. | Produce a proposal only; no record is merged by this request. |

| POST /v1/issues/summarize | Summarize a collection or theme. | Require source IDs and return a traceable, bounded summary. |

| POST /v1/developer/brief | Generate a developer-facing problem brief. | Separate evidence, assumptions, unknowns, and follow-up questions. |

Every endpoint should have an explicit versioned Pydantic schema, request-size limit, timeout, authentication/authorization policy, and predictable error format. The exact routes can change during implementation, but the API must make the approval boundary unmistakable.


### Illustrative comparison response

```json
{
  "candidate_issue_id": "issue_123",
  "relationship": "related_but_distinct",
  "confidence": 0.78,
  "evidence": ["Both reports involve slow project loading"],
  "differences": ["The candidate describes a delay; the new report describes a complete freeze"],
  "recommended_action": "human_review",
  "auto_merge": false,
  "model_version": "mendseed-issue-assistant-v0.1"
}
```

The scores above are illustrative only; they are not calibrated model output or recommended production thresholds. Calibrate thresholds on held-out examples. A score must not be presented as a probability unless it has been evaluated and calibrated as one.


## 10. Evaluation and release gates

Build a “golden set” of at least 100 held-out cases for the first meaningful comparison, expanding it as real reviewer outcomes become available. The set should include ordinary, ambiguous, adversarial, and privacy-sensitive examples. Keep a fixed baseline so improvements can be attributed to changes.

| Area | Measure | Release question |

| --- | --- | --- |

| Output contract | Valid JSON and schema pass rate; missing/extra keys. | Does the service return safely parseable structured data? |

| Factuality | Unsupported-claim rate and reviewer-rated faithfulness. | Does the model add facts that the user did not provide? |

| Extraction | Per-field accuracy/completeness and unknown handling. | Does it preserve what is known and identify what is missing? |

| Retrieval | Recall@5/Recall@10 for known related and duplicate pairs. | Are useful candidates included in the retrieved set? |

| Comparison | Precision/recall/F1 by relationship label; confusion matrix. | Can it distinguish duplicates from related-but-distinct issues? |

| Human workflow | Reviewer acceptance rate, reversal rate, time saved. | Does it help people make decisions without removing control? |

| Operations | Latency percentiles, errors, memory, cost per request. | Can the service run within the project’s real budget and reliability needs? |

| Privacy/security | Sensitive-data handling tests, access-control tests, log scans. | Are sensitive values minimized and kept out of routine logs? |

Provisional release gates: schema validation should be extremely reliable (aim for at least 99% on the held-out set); no automatic merge path should exist; high-confidence duplicate suggestions must meet a precision target agreed after measurement (a provisional 0.90+ is a starting target, not a claim); critical privacy failures block release. Report sample size and uncertainty with every metric.


## 11. Deployment and privacy model


### Notebook is for experimentation, not production hosting

The current Google Colab T4 is useful for testing and fine-tuning, but notebook sessions can reset or expire and are not a production API. Save checkpoints, adapters, dataset manifests, and evaluation results to durable storage, with access controls. Pin package and model revisions for reproducibility.


### Production path

1. Package inference behind a FastAPI service with stable versioned schemas.
2. Choose a hosting option only after measuring memory, throughput, tail latency, and request volume: a managed inference service or an appropriately sized GPU server are both possible.
3. Keep the API and inference worker separable so requests can be queued, timed out, retried safely, or moved to another model later.
4. Add authentication, rate limits, quotas, request-size limits, health checks, monitoring, secret management, and deploy/rollback procedures.
5. Version the base model, adapter, prompts, schemas, embedding model, and index generation. A release should be reproducible and reversible.


### Privacy principles to implement

- Data minimization: send only fields required for the requested task.
- Redaction: remove likely secrets and unnecessary personal data before inference where possible.
- No sensitive payloads in logs by default; record request IDs, timings, error codes, and version metadata instead.
- Access control and retention rules for raw reports, embeddings, labels, and training exports.
- Clear policy for consent, deletion, user correction, training-data inclusion, and third-party inference providers.
- No claim of “privacy-first” based solely on the model choice; privacy must be demonstrated by the whole data flow.


## 12. Implementation roadmap: what, why, how, and when

| Phase | When / sequence | Work | Exit condition |

| --- | --- | --- | --- |

| 0. Define the contract | Now; before training | Finalize fields, relationship labels, policy boundaries, JSON schemas, sample cases, privacy assumptions. | Written schema and initial golden tests exist. |

| 1. Prove base inference | Immediate next step | Load the chosen model in 4-bit on Colab T4; run one deterministic extraction test; verify output, GPU memory, and actual generation. | Model loads and produces a response reliably in the current runtime. |

| 2. Baseline evaluation | After the smoke test | Create 100+ representative held-out examples; run the untouched model; record schema/factuality/task metrics. | Baseline report identifies measurable weaknesses. |

| 3. Build reviewed data | Before first fine-tune | Label extraction examples and comparison pairs; scrub data; version and split by issue group. | Training and evaluation sets are versioned and leakage-checked. |

| 4. Fine-tune adapter | Only when baseline/data justify it | Run a short QLoRA/LoRA SFT experiment; save adapter and config; track loss and runtime. | Adapter loads cleanly and training run is reproducible. |

| 5. Compare and select | Immediately after training | Run baseline and adapter against the same golden set; review failures and regressions. | Keep adapter only if measured improvements are meaningful. |

| 6. Build AI pipeline | After model behavior is acceptable | Add FastAPI, schema validation, embedding retrieval, comparison layer, PostgreSQL/pgvector, review proposals. | End-to-end flow returns proposals without autonomous record changes. |

| 7. Staging integration | After local tests | Connect the MendSeed app to staging; add auth, rate limits, metrics, timeout behavior, audit events. | Human review flow works end to end and can be rolled back. |

| 8. Limited release | After gates pass | Release to a small opt-in group; collect reviewer corrections; monitor metrics and privacy events. | Quality and operational behavior meet agreed targets. |

This is a sequence, not a promise of calendar duration. Timing depends on data-labeling effort, Colab availability, actual model compatibility, and deployment budget. Do not skip evaluation or privacy design to meet an arbitrary date.


## 13. Current status: established facts versus unfinished work

| Item | Status | Interpretation |

| --- | --- | --- |

| Product purpose and scope | Defined at planning level | MendSeed AI is issue intelligence and decision support, not a general chatbot. |

| Training environment | Available in Google Colab | Python 3.13.15, PyTorch 2.11.0+cu130, CUDA available, Tesla T4 with about 14.56 GiB reported VRAM. |

| Libraries | Installed / import and config check reported | Transformers, Accelerate, Datasets, PEFT, TRL, bitsandbytes were installed; bitsandbytes 0.50.2 and config construction worked. |

| CUDA 4-bit kernel path | Not yet proven by config creation alone | Actual model loading and generation are the next smoke test. |

| Base model | Selected as first candidate | Qwen/Qwen3-4B-Instruct-2507; not yet declared final. |

| Dataset and labels | To be created/reviewed | No assumption that a production-quality labeled corpus already exists. |

| Fine-tuning | Not yet confirmed complete | Do not call the system trained until an adapter is saved and evaluated. |

| Hosted API | Planned | The notebook is not a reliable production server; hosting and budget remain to be decided. |

Immediate next action: run the model-load and single-inference smoke test in the current T4 notebook. If it succeeds, record load time, GPU memory, generated output, and any warnings; then establish a fixed baseline evaluation. If it fails, capture the complete error before changing the Torch/CUDA stack.


## 14. Key risks and mitigations

| Risk | Why it matters | Mitigation |

| --- | --- | --- |

| False duplicate merges | Distinct problems can be lost or misrepresented. | No autonomous merges; conservative thresholds; explicit differences; easy rejection; audit trail. |

| Hallucinated fields | Invented causes, severity, reproduction steps, or impact undermine trust. | Unknown values; schema validation; evidence separation; held-out factuality tests. |

| Small or biased dataset | The adapter may overfit and work poorly on real community text. | Reviewed diverse examples, group splits, failure-driven collection, baseline comparison. |

| Training/evaluation leakage | Reported performance may be misleading. | Split by issue family/source/time; deduplicate before splitting. |

| Privacy leakage | Issue reports may contain personal details or secrets. | Redaction, access control, retention rules, minimal logs, opt-in data use. |

| Notebook reset or dependency drift | Experiments become unreproducible. | Save artifacts externally, pin versions, store configuration and run manifests. |

| Hosting cost/latency | A 4B model may be too expensive or slow for the first product iteration. | Benchmark real workloads; async jobs; cache safe results; test smaller models and managed hosting options. |

| Overreliance on model-generated summaries | Themes can sound certain despite sparse evidence. | Link source reports, show sample sizes, label hypotheses, allow users to inspect evidence. |


## 15. Definition of a successful first release

The first release is successful when it can take a raw report, produce reliable schema-valid structured fields, find plausible matching reports, explain similarities and differences, and present a human-reviewable proposal without changing canonical records on its own. It should be possible to reproduce the model/version used for each proposal and measure performance against a held-out test set.

- A working base-model baseline and an evaluation report exist.
- The selected model/adapter revision and all prompts/schemas are versioned.
- Output validation and safe fallback behavior work even for malformed model responses.
- Duplicate retrieval and relationship classification are measured separately.
- Human review and audit history are integrated into the product flow.
- Privacy, auth, rate limits, and deployment rollback are implemented before public exposure.
- The system’s quality claims are supported by reported test results rather than intuition.


## 16. Decisions still required

- Which issue fields are canonical and which are optional in the MendSeed product schema?
- What are the precise definitions and reviewer instructions for duplicate versus related-but-distinct?
- Will issue text be sent to a managed inference provider, or must production inference run on infrastructure controlled by the project?
- What are the expected request volume, latency target, monthly budget, and data-retention requirements?
- Which language(s) and report types must be supported in version 1?
- Who is authorized to approve merges, moderation flags, and public theme summaries?
- Will reviewed corrections be used for future training, and what consent, deletion, and provenance process governs that use?


## 17. Reference links

Model card: Qwen/Qwen3-4B-Instruct-2507 — https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

Hugging Face PEFT quantization / QLoRA guide — https://huggingface.co/docs/peft/developer_guides/quantization

Hugging Face TRL SFTTrainer documentation — https://huggingface.co/docs/trl/sft_trainer

Hugging Face TRL PEFT integration — https://huggingface.co/docs/trl/peft_integration
