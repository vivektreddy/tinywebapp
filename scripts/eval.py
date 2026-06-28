"""Eval script: scores retrieval recall and response quality via LLM-as-judge."""
import json, sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from rag_search import RAGSearcher
from config import settings

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful concise assistant to help people in California who just lost their jobs. "
    "Base your answers on the retrieved documents. Cite sources inline using bracketed numbers like [1] or [2]. "
    "If no documents pertain to the user's question, use your general knowledge and say so. "
    "Do not answer questions unrelated to helping user. "
    "Here are the retrieved documents:\n\n{context}"
)

JUDGE_SYSTEM = (
    "You are an evaluator for a California benefits chatbot. "
    "Score the response on three dimensions (1-5 each):\n"
    "- relevance: Does the response directly answer the question asked?\n"
    "- accuracy: Is the information factually correct based on the retrieved sources?\n"
    "- completeness: Does it cover the main aspects of the question?\n\n"
    'Return ONLY valid JSON: {"relevance": N, "accuracy": N, "completeness": N, "reasoning": "..."}'
)


def score_retrieval(retrieved_urls: list, expected_urls: list):
    if not expected_urls:
        return None
    found = sum(1 for u in expected_urls if u in retrieved_urls)
    return found / len(expected_urls)


def get_response(question: str, context: list) -> str:
    client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    numbered = "\n\n".join(
        f"[{i+1}] Title: {d['title']}\nURL: {d['url']}\n{d['excerpt']}"
        for i, d in enumerate(context)
    )
    resp = client.converse(
        modelId=settings.DEFAULT_MODEL.value,
        messages=[{"role": "user", "content": [{"text": question}]}],
        system=[{"text": SYSTEM_PROMPT_TEMPLATE.format(context=numbered)}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
    )
    return resp["output"]["message"]["content"][0]["text"]


def judge_response(question: str, response_text: str, context: list) -> dict:
    client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    sources_summary = "\n".join(f"- {d['title']} ({d['url']})" for d in context)
    judge_input = (
        f"Question: {question}\n\n"
        f"Retrieved sources:\n{sources_summary}\n\n"
        f"Response to evaluate:\n{response_text}"
    )
    resp = client.converse(
        modelId=settings.DEFAULT_MODEL.value,
        messages=[{"role": "user", "content": [{"text": judge_input}]}],
        system=[{"text": JUDGE_SYSTEM}],
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},
    )
    try:
        return json.loads(resp["output"]["message"]["content"][0]["text"])
    except Exception:
        return {"relevance": 0, "accuracy": 0, "completeness": 0, "reasoning": "parse error"}


def main():
    questions = json.loads(EVAL_SET.read_text())
    searcher = RAGSearcher()
    results = []

    print(f"{'ID':<25} {'Recall':>7} {'Rel':>5} {'Acc':>5} {'Comp':>6}")
    print("-" * 55)

    for q in questions:
        context = searcher.search(q["question"])
        retrieved_urls = [d["url"] for d in context]
        recall = score_retrieval(retrieved_urls, q.get("expected_urls", []))
        response_text = get_response(q["question"], context)
        scores = judge_response(q["question"], response_text, context)

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "retrieval_recall": recall,
            "retrieved_urls": retrieved_urls,
            "judge_scores": scores,
            "response": response_text,
        })

        recall_str = f"{recall:.0%}" if recall is not None else "  N/A"
        print(f"{q['id']:<25} {recall_str:>7} {scores['relevance']:>5} {scores['accuracy']:>5} {scores['completeness']:>6}")

    scoreable = [r for r in results if r["retrieval_recall"] is not None]
    avg_recall = sum(r["retrieval_recall"] for r in scoreable) / len(scoreable) if scoreable else 0
    avg_rel = sum(r["judge_scores"]["relevance"] for r in results) / len(results)
    avg_acc = sum(r["judge_scores"]["accuracy"] for r in results) / len(results)
    avg_comp = sum(r["judge_scores"]["completeness"] for r in results) / len(results)

    print("-" * 55)
    print(f"{'AVERAGE':<25} {avg_recall:>7.0%} {avg_rel:>5.1f} {avg_acc:>5.1f} {avg_comp:>6.1f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_results_{timestamp}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
