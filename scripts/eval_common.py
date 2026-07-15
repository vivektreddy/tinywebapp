"""Shared constants and helpers for eval.py and eval_e2e.py."""
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import boto3
from config import settings

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant for people in California who just lost their jobs. "
    "Be warm, clear, and concise. Users are stressed — get to the point quickly and avoid jargon. "
    "Cite sources inline using bracketed numbers like [1] or [2] based on the retrieved documents. "
    "If the retrieved documents do not contain relevant information, say so plainly and suggest the user "
    "call 211 or visit their county social services office for personalized help. "
    "If the question is clearly unrelated to job loss or financial hardship (e.g. restaurant recommendations), "
    "decline politely and redirect to how you can help. "
    "Here are the retrieved documents:\n\n{context}"
)

E2E_SYSTEM_PROMPT = (
    "You are a helpful assistant for people in California who just lost their jobs. "
    "Be warm, clear, and concise. Users are stressed — get to the point quickly and avoid jargon. "
    "Use the rag_search tool whenever the user asks about California benefits, programs, eligibility, "
    "how to apply, or anything related to unemployment, food assistance, healthcare, housing, or job training. "
    "Also use it for adjacent topics like resume help or interview tips — these users need practical guidance. "
    "Cite sources inline using bracketed numbers like [1] or [2] based on the search results. "
    "If the search results do not contain relevant information, say so plainly and suggest the user "
    "call 211 or visit their county social services office for personalized help. "
    "If the question is clearly unrelated to job loss or financial hardship (e.g. restaurant recommendations), "
    "decline politely and redirect to how you can help."
)

TOOL_CONFIG = {
    "tools": [{"toolSpec": {
        "name": "rag_search",
        "description": (
            "Search the California benefits knowledge base. Use this when the user asks about "
            "California government benefits, programs, eligibility, or how to apply for assistance "
            "such as CalFresh, unemployment insurance, Medi-Cal, CalWORKs, WIC, SDI, PFL, "
            "CARE/FERA utilities discount, or job training programs."
        ),
        "inputSchema": {"json": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }},
    }}],
    "toolChoice": {"auto": {}},
}

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
