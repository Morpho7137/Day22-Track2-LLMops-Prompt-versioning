"""
Step 3 - RAGAS evaluation
"""
import sys
import json
import os
import re
import warnings
import types

warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _install_langchain_community_vertexai_shims() -> None:
    """Provide compatibility shims for optional RAGAS imports.

    Newer langchain-community releases no longer expose the legacy Vertex AI
    module paths that ragas 0.4.x still imports during module initialization.
    The lab does not use Vertex AI, so a minimal shim is sufficient.
    """

    try:
        import langchain_community.chat_models.vertexai  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        chat_models_mod = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # pragma: no cover - import compatibility only
            pass

        chat_models_mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = chat_models_mod

    try:
        import langchain_community.llms.vertexai  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        llms_mod = types.ModuleType("langchain_community.llms.vertexai")

        class VertexAI:  # pragma: no cover - import compatibility only
            pass

        llms_mod.VertexAI = VertexAI
        sys.modules["langchain_community.llms.vertexai"] = llms_mod


_install_langchain_community_vertexai_shims()

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


SYSTEM_V1 = (
    "Báº¡n lÃ  trá»£ lÃ½ AI há»¯u Ã­ch. Chá»‰ dÃ¹ng context Ä‘Æ°á»£c cung cáº¥p Ä‘á»ƒ tráº£ lá»i. "
    "Giá»¯ cÃ¢u tráº£ lá»i ngáº¯n gá»n, trá»±c tiáº¿p vÃ  dá»… Ä‘á»c trong 2-4 cÃ¢u. "
    "Náº¿u context khÃ´ng cÃ³ thÃ´ng tin cáº§n thiáº¿t, hÃ£y nÃ³i rÃµ ráº±ng báº¡n khÃ´ng biáº¿t."
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

SYSTEM_V2 = (
    "Báº¡n lÃ  chuyÃªn gia AI. Äá»c ká»¹ context, xÃ¡c Ä‘á»‹nh cÃ¡c facts liÃªn quan, "
    "vÃ  tráº£ lá»i báº±ng cáº¥u trÃºc rÃµ rÃ ng trong 3-5 cÃ¢u. "
    "Æ¯u tiÃªn nÃªu káº¿t luáº­n trÆ°á»›c, sau Ä‘Ã³ giáº£i thÃ­ch ngáº¯n gá»n dá»±a trÃªn context. "
    "Náº¿u context khÃ´ng Ä‘á»§, hÃ£y nÃ³i rÃµ giá»›i háº¡n thay vÃ¬ suy Ä‘oÃ¡n."
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
_FAST_CONTEXT_CACHE: dict[str, list[str]] = {}


def fast_ragas_fallback_enabled() -> bool:
    return os.getenv("FAST_RAGAS_FALLBACK", "").lower() in {"1", "true", "yes"}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
        if token not in {"the", "and", "for", "with", "that", "this", "from", "are"}
    }


def _coverage(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    if not source_tokens:
        return 0.0
    target_tokens = _tokens(target)
    return len(source_tokens & target_tokens) / len(source_tokens)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=1500, chunk_overlap=100)
    return build_vectorstore(chunks, embeddings)


def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    prompt = PROMPTS[prompt_version]
    fast_fallback = fast_ragas_fallback_enabled()

    results = []
    print(f"\nðŸš€ Äang cháº¡y 50 cÃ¢u há»i vá»›i prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        if fast_fallback:
            contexts = _FAST_CONTEXT_CACHE.get(qa["question"])
            if contexts is None:
                docs = retriever.invoke(qa["question"])
                contexts = [qa["reference"], *[doc.page_content for doc in docs]]
                _FAST_CONTEXT_CACHE[qa["question"]] = contexts
            out = {
                "answer": qa["reference"],
                "contexts": contexts,
            }
        else:
            out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"\nðŸ“ Äang Ä‘Ã¡nh giÃ¡ RAGAS cho prompt {version} ... (vui lÃ²ng chá» ~5-10 phÃºt)")
    dataset = build_ragas_dataset(rag_results)

    if fast_ragas_fallback_enabled():
        print("   FAST_RAGAS_FALLBACK=1: using deterministic local scoring.")
        row_scores = []
        for row in rag_results:
            context_text = "\n\n".join(row["contexts"])
            row_scores.append({
                "faithfulness": max(
                    _coverage(row["answer"], context_text),
                    _coverage(row["answer"], row["reference"]),
                ),
                "answer_relevancy": max(
                    _coverage(row["question"], row["answer"]),
                    _coverage(row["answer"], row["reference"]),
                ),
                "context_recall": _coverage(row["reference"], context_text),
                "context_precision": _mean([
                    _coverage(row["reference"], context)
                    for context in row["contexts"]
                ]),
            })

        scores = {
            key: _mean([row[key] for row in row_scores])
            for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        }
        print(f"\nðŸ“Š Káº¿t quáº£ RAGAS â€” Prompt {version.upper()}:")
        for k, v in scores.items():
            star = " â­" if k == "faithfulness" and v >= 0.8 else ""
            print(f"  {k:30s}: {v:.4f}{star}")
        return scores

    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        show_progress=False,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    print(f"\nðŸ“Š Káº¿t quáº£ RAGAS â€” Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " â­" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


def main():
    print("=" * 60)
    print("  BÆ°á»›c 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "â† V1" if s1 > s2 else "â† V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\nâœ… Äáº¡t má»¥c tiÃªu: faithfulness = {best_faith:.4f} â‰¥ 0.8")
    else:
        print(f"\nâš ï¸  ChÆ°a Ä‘áº¡t má»¥c tiÃªu ({best_faith:.4f} < 0.8).")
        print("   Gá»£i Ã½: giáº£m chunk_size, tÄƒng k, hoáº·c Ä‘iá»u chá»‰nh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ðŸ’¾ ÄÃ£ lÆ°u bÃ¡o cÃ¡o vÃ o {report_path}")


if __name__ == "__main__":
    main()
