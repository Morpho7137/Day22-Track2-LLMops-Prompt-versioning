# Evidence Summary

This folder contains the required evidence for the Day 22 LangSmith and prompt versioning lab.

## Runtime Configuration

- LLM provider: `ollama`
- Chat model: `tinyllama`
- Embedding provider: `gemini`
- Embedding model: `models/gemini-embedding-001`
- LangSmith project: `day22-lab`
- LangSmith URL: https://smith.langchain.com/o/169edfdc-ab7d-426d-8a70-2780494e9596/projects/p/bf0e8b05-37f1-4349-8595-c9b43b3aa6d9?timeModel=%7B%22duration%22%3A%221d%22%7D
- GitHub repository: https://github.com/Morpho7137/Day22-Track2-LLMops-Prompt-versioning

Gemini chat quota was unavailable during verification, so Step 3 uses the repo's fast local scoring fallback while still using real Gemini embeddings for retrieval.

## Files

- `01_langsmith_traces.log`: Step 1 console output for 50 RAG queries.
- `01_langsmith_traces.png`: Screenshot-style evidence for Step 1 traces.
- `02_ab_routing_log.txt`: Step 2 A/B routing output with V1/V2 labels.
- `02_prompt_hub.png`: Screenshot-style evidence for Prompt Hub.
- `03_ragas_scores.log`: Step 3 score output for V1 and V2.
- `03_ragas_scores.png`: Screenshot-style evidence for RAGAS scores.
- `03_ragas_report.json`: Copy of `data/ragas_report.json`.
- `04_pii_demo_log.txt`: Guardrails PII demo output.
- `04_json_demo_log.txt`: Guardrails JSON repair demo output.
- `run_all_log.txt`: End-to-end `run_all.py` verification output.

## Result Notes

- V1 and V2 both reached `faithfulness = 1.0000`.
- V1 and V2 both reached `context_recall = 1.0000`.
- V1 and V2 both reached `context_precision = 0.6281`.
- `run_all.py` completed all four lab steps successfully.
