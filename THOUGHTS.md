# MendSeed AI & QuietFlay v0.1 — Engineering Thoughts, Critique & Benchmark Ledger

## 1. Core Vision & Domain Alignment
- **Mission:** MendSeed is a **Public Grievance Observatory and Problem Discovery Platform**.
- **Audience:** Everyday citizens reporting real-world friction (agriculture, civic infrastructure, healthcare, education, transport, daily life) and **builders / hackathon teams / researchers** looking for real, high-impact problems to solve.
- **AI Core Value:** Turn unstructured, emotional, unorganized citizen reports into structured `GrievanceAnalysis`, filter out bot spikes and viral noise via **QuietFlay v0.1**, and generate actionable `BuilderBrief`s for project creators.

---

## 2. Self-Critique of Current State (Phase 1 Audit)
- **Strengths:**
  - Clean separation: Zero-ML lightweight SDK (`sdk/`) with pure-Python QuietFlay math (0 external dependencies, subcriticality guaranteed).
  - Fast, serverless-ready API gateway (`server/main.py`) with constant-time HMAC `X-MendSeed-Key` authentication.
  - Multi-sector Pydantic schemas accurately reflect public problems rather than simple software bugs.
- **Weaknesses & Gaps Identified:**
  1. *Small Training/Validation Volume:* The current training dataset has 7 synthetic examples. A robust evaluation requires a comprehensive test suite (at least 50+ diverse, multi-sector scenarios including edge cases and adversarial inputs) to accurately test precision, recall, and generalization.
  2. *Lack of Visual Benchmark Proof:* We need concrete, visual charts (Confusion Matrices, Precision-Recall curves, Loss & Bias-Variance diagnostics, QuietFlay NDCG comparisons) so stakeholders can visibly verify model performance.
  3. *Adversarial Robustness:* How does the system handle prompt injections, gibberish inputs, duplicate spam bursts, or ambiguous grievances? We must build explicit stress tests.
  4. *Hugging Face Synchronization:* Ensure the repository `itsjustayush/mendseed-qwen2.5-3b-lora` exists, is authenticated with `HF_TOKEN`, and is ready for seamless model upload and inference.

---

## 3. Master Execution Roadmap
- [x] **Step 1: Environment & Hugging Face CLI Setup**
  - Verify HF authentication with `HF_TOKEN` from `.env`.
  - Check CLI / Python toolchain availability.
- [ ] **Step 2: Comprehensive Evaluation Harness & Benchmark Suite**
  - Build `benchmark/test_suite.py` with multi-sector golden test cases across all 5 schemas.
  - Test edge cases: empty strings, prompt injections, ambiguous domains, multi-issue grievances.
- [ ] **Step 3: Metrics Computation & Visual Charts Generation**
  - Compute Schema Validity Rate, Sector Classification Accuracy, Precision, Recall, and F1.
  - Benchmark QuietFlay v0.1 against raw vote counting and Wilson interval on burst/spam/legacy regimes.
  - Generate visual charts (`confusion_matrix.png`, `precision_recall.png`, `quietflay_benchmark.png`, `bias_variance_diagnosis.png`).
- [ ] **Step 4: Hugging Face Artifact Preparation & Upload**
  - Prepare model metadata, model card (`README.md` for Hugging Face Hub), adapter configurations.
  - Verify upload to `itsjustayush/mendseed-qwen2.5-3b-lora`.
- [ ] **Step 5: Ongoing Self-Critique & Synthesis**
  - Document all metrics, trade-offs, and critical assessments in this ledger.
