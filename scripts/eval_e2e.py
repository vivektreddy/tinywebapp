"""End-to-end eval: runs the full agentic tool use loop identical to app.py.

Measures tool invocation correctness (did Claude call/skip the tool appropriately?)
in addition to retrieval recall and response quality.

Usage:
  python scripts/eval_e2e.py
"""
import json, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from rag_search import RAGSearcher
from config import settings
from eval_common import (
    EVAL_SET, RESULTS_DIR, E2E_SYSTEM_PROMPT, TOOL_CONFIG,
    score_retrieval, judge_response,
)


def get_response_e2e(question: str, searcher: RAGSearcher) -> dict:
    client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
    messages = [{"role": "user", "content": [{"text": question}]}]

    phase1 = client.converse(
        modelId=settings.DEFAULT_MODEL.value,
        messages=messages,
        system=[{"text": E2E_SYSTEM_PROMPT}],
        toolConfig=TOOL_CONFIG,
        inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
    )

    tool_called = phase1["stopReason"] == "tool_use"
    query_used = None
    context = []

    if tool_called:
        tool_block = next(b["toolUse"] for b in phase1["output"]["message"]["content"] if "toolUse" in b)
        query_used = tool_block["input"]["query"]
        tool_use_id = tool_block["toolUseId"]

        context = searcher.search(search_query=query_used)
        numbered = "\n\n".join(
            f"[{i+1}] Title: {d['title']}\nURL: {d['url']}\n{d['excerpt']}"
            for i, d in enumerate(context)
        )

        messages = messages + [
            {"role": "assistant", "content": phase1["output"]["message"]["content"]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": numbered}]}}]},
        ]
        phase2 = client.converse(
            modelId=settings.DEFAULT_MODEL.value,
            messages=messages,
            system=[{"text": E2E_SYSTEM_PROMPT}],
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
        )
        response_text = phase2["output"]["message"]["content"][0]["text"]
    else:
        response_text = phase1["output"]["message"]["content"][0]["text"]

    return {
        "response_text": response_text,
        "tool_called": tool_called,
        "query_used": query_used,
        "context": context,
    }


def main():
    questions = json.loads(EVAL_SET.read_text())
    searcher = RAGSearcher()
    results = []

    print(f"{'ID':<25} {'Tool?':>6} {'Recall':>7} {'Rel':>5} {'Acc':>5} {'Comp':>6}")
    print("-" * 62)

    for q in questions:
        expects_tool = bool(q.get("expected_urls"))
        e2e = get_response_e2e(q["question"], searcher)

        retrieved_urls = [d["url"] for d in e2e["context"]]
        recall = score_retrieval(retrieved_urls, q.get("expected_urls", []))
        scores = judge_response(q["question"], e2e["response_text"], e2e["context"])
        tool_correct = e2e["tool_called"] == expects_tool

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "tool_called": e2e["tool_called"],
            "tool_correct": tool_correct,
            "query_used": e2e["query_used"],
            "retrieval_recall": recall,
            "retrieved_urls": retrieved_urls,
            "judge_scores": scores,
            "response": e2e["response_text"],
        })

        recall_str = f"{recall:.0%}" if recall is not None else "  N/A"
        tool_str = ("Y" if e2e["tool_called"] else "N") + ("" if tool_correct else "!")
        print(f"{q['id']:<25} {tool_str:>6} {recall_str:>7} {scores['relevance']:>5} {scores['accuracy']:>5} {scores['completeness']:>6}")

    scoreable = [r for r in results if r["retrieval_recall"] is not None]
    avg_recall = sum(r["retrieval_recall"] for r in scoreable) / len(scoreable) if scoreable else 0
    avg_rel = sum(r["judge_scores"]["relevance"] for r in results) / len(results)
    avg_acc = sum(r["judge_scores"]["accuracy"] for r in results) / len(results)
    avg_comp = sum(r["judge_scores"]["completeness"] for r in results) / len(results)
    tool_accuracy = sum(1 for r in results if r["tool_correct"]) / len(results)

    print("-" * 62)
    print(f"{'AVERAGE':<25} {tool_accuracy:>6.0%} {avg_recall:>7.0%} {avg_rel:>5.1f} {avg_acc:>5.1f} {avg_comp:>6.1f}")
    print("(Tool? column: Y=called, N=skipped, !=wrong decision)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_results_e2e_{timestamp}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
