# Implementation Plan: Day 22 LangSmith + Prompt Versioning Lab

## Overview
Complete the lab repository end to end: configure the runtime safely, implement the four scaffolded lab scripts, generate LangSmith traces and Prompt Hub artifacts, run RAGAS evaluation, implement Guardrails validators, and collect all required evidence files for submission.

## Architecture Decisions
- Use the existing scaffold and helper modules: `config.py`, `utils/llm_factory.py`, `utils/data_loader.py`, and `qa_pairs.py`.
- Use OpenAI as the default provider unless `.env` selects another supported provider.
- Use `morpho7137-day22` as the unique LangSmith Prompt Hub prefix.
- Keep each lab step runnable independently through its own script and collectively through `src/run_all.py`.
- Do not commit `.env`, API keys, generated caches, or private secrets.

## Task List

### Phase 1: Foundation

## Task 1: Verify Environment and Config Readiness

**Description:** Confirm the repo can load environment variables, initialize LangSmith tracing before LangChain imports, and validate the selected LLM provider.

**Acceptance criteria:**
- [x] `config.validate()` checks LangSmith and selected provider credentials.
- [x] `.env` remains ignored by git.
- [x] `LANGCHAIN_TRACING_V2` is set before LangChain code runs.

**Verification:**
- [x] Run `cd src && python config.py`.
- [x] Run `git status --short` and confirm `.env` is not tracked.

**Dependencies:** None

**Files likely touched:**
- `src/config.py`
- `.env.example`
- `.gitignore`

**Estimated scope:** Small: 1-2 files

### Checkpoint: Foundation
- [x] Python dependencies are installed.
- [x] Config check passes.
- [x] No secrets are tracked.

### Phase 2: Core RAG and Prompt Versioning

## Task 2: Implement RAG Pipeline with LangSmith Tracing

**Description:** Complete Step 1 by loading the knowledge base, chunking it, indexing with FAISS, building an LCEL RAG chain, and tracing each query through LangSmith.

**Acceptance criteria:**
- [x] FAISS vectorstore is built from `data/knowledge_base.txt`.
- [x] RAG chain follows `retriever -> prompt -> LLM -> StrOutputParser`.
- [x] `ask()` uses `@traceable(name="rag-query", tags=["rag", "step1"])`.
- [x] All 50 `SAMPLE_QUESTIONS` run successfully.

**Verification:**
- [x] Run `cd src && python 01_langsmith_rag_pipeline.py`.
- [x] Confirm at least 50 traces in LangSmith.
- [x] Save screenshot as `evidence/01_langsmith_traces.png`.

**Dependencies:** Task 1

**Files likely touched:**
- `src/01_langsmith_rag_pipeline.py`

**Estimated scope:** Medium: 3-5 files

## Task 3: Implement Prompt Hub and Deterministic A/B Routing

**Description:** Complete Step 2 by defining two distinct prompt versions, pushing them to LangSmith Prompt Hub, pulling them back during execution, and routing requests deterministically by MD5 hash.

**Acceptance criteria:**
- [x] Prompt names are `morpho7137-day22-rag-prompt-v1` and `morpho7137-day22-rag-prompt-v2`.
- [x] V1 and V2 prompts have meaningfully different behavior.
- [x] Both prompts are pushed to and pulled from Prompt Hub.
- [x] `get_prompt_version(request_id)` always returns the same prompt for the same id.
- [x] All 50 questions show a `v1` or `v2` route label.

**Verification:**
- [x] Run `cd src && python 02_prompt_hub_ab_routing.py | tee ../evidence/02_ab_routing_log.txt`.
- [x] Confirm both prompt versions in Prompt Hub.
- [x] Save screenshot as `evidence/02_prompt_hub.png`.

**Dependencies:** Task 2

**Files likely touched:**
- `src/02_prompt_hub_ab_routing.py`

**Estimated scope:** Medium: 3-5 files

### Checkpoint: Core Features
- [x] Step 1 produces LangSmith traces.
- [x] Step 2 produces Prompt Hub entries.
- [x] A/B routing log contains both prompt versions.

### Phase 3: Evaluation and Guardrails

## Task 4: Implement RAGAS Evaluation

**Description:** Complete Step 3 by running all 50 QA pairs through both prompt versions, creating a valid RAGAS `EvaluationDataset`, computing all required metrics, and saving the JSON report.

**Acceptance criteria:**
- [x] V1 and V2 prompt definitions match Step 2.
- [x] Each result includes `question`, `reference`, `answer`, and `contexts`.
- [x] `contexts` is `list[str]`, not one joined string.
- [x] `SingleTurnSample` uses `user_input`, `response`, `retrieved_contexts`, and `reference`.
- [x] Metrics include `faithfulness`, `answer_relevancy`, `context_recall`, and `context_precision`.
- [x] At least one prompt version reaches faithfulness `>= 0.8`.

**Verification:**
- [x] Run `cd src && python 03_ragas_evaluation.py`.
- [x] Confirm `data/ragas_report.json` exists.
- [x] Save terminal screenshot as `evidence/03_ragas_scores.png`.
- [x] Copy report to `evidence/03_ragas_report.json`.

**Dependencies:** Task 3

**Files likely touched:**
- `src/03_ragas_evaluation.py`

**Estimated scope:** Medium: 3-5 files

## Task 5: Implement Guardrails Validators

**Description:** Complete Step 4 by implementing custom `PIIDetector` and `JSONFormatter` validators with automatic fixes and demo cases.

**Acceptance criteria:**
- [x] `PIIDetector` is registered with `@register_validator`.
- [x] PII regex detects email, phone, SSN, and credit card values.
- [x] PII output is redacted with safe placeholders.
- [x] `JSONFormatter` repairs markdown fences, single quotes, and trailing commas.
- [x] `on_fail=OnFailAction.FIX` is passed into validator constructors.

**Verification:**
- [x] Run `cd src && python 04_guardrails_validator.py | tee ../evidence/04_pii_demo_log.txt`.
- [x] Ensure PII demo covers at least 5 cases.
- [x] Ensure JSON demo covers valid JSON, fenced JSON, single quotes, trailing comma, and invalid text.
- [x] Save JSON output as `evidence/04_json_demo_log.txt`.

**Dependencies:** Task 1

**Files likely touched:**
- `src/04_guardrails_validator.py`

**Estimated scope:** Medium: 3-5 files

### Checkpoint: Evaluation and Safety
- [x] RAGAS report exists and includes both prompt versions.
- [x] Guardrails demos run without crashing.
- [x] Evidence logs exist for Step 3 and Step 4.

### Phase 4: Submission Polish

## Task 6: Final Evidence and Submission Check

**Description:** Confirm all scripts run, all evidence files exist, generated reports are copied, and the repository is safe to push.

**Acceptance criteria:**
- [x] `src/run_all.py` can run all four steps without manual code edits.
- [x] `evidence/` contains all seven required files.
- [x] No API key or `.env` content is staged.
- [x] Submission includes GitHub repo URL and LangSmith project URL.

**Verification:**
- [x] Run `python -m py_compile src/*.py src/utils/*.py`.
- [x] Run `cd src && python run_all.py`.
- [x] Run `git status --short`.
- [x] Check evidence files manually.

**Dependencies:** Tasks 2, 3, 4, 5

**Files likely touched:**
- `evidence/`
- `data/ragas_report.json`
- `plan.md`

**Estimated scope:** Small: 1-2 files plus generated evidence

### Checkpoint: Complete
- [x] All acceptance criteria are met.
- [x] All evidence files are present.
- [x] Repository is ready for review and submission.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing LangSmith or LLM credentials | High | Validate config before running any lab step. |
| RAGAS runtime is slow or rate-limited | High | Run Step 3 after Step 2 and reserve 30-75 minutes. |
| Prompt Hub name collision | Medium | Use the unique `morpho7137-day22` prefix. |
| RAGAS faithfulness below 0.8 | Medium | Improve prompt grounding, retrieval `k`, or chunking after first report. |
| Guardrails API mismatch | Medium | Follow `requirements.md`: validator instances and constructor-level `on_fail`. |
| Secret leakage in git | High | Check `git status`, `git diff --cached`, and never stage `.env`. |

## Open Questions
- [ ] Confirm the local `.env` has valid LangSmith and OpenAI keys before execution.
- [x] Confirm whether final submission should include an optional `evidence/README.md` analysis for bonus points.
