import os, json, logging, time
import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import boto3
from schemas import ChatRequest
from uuid import uuid4
from config import settings
from rag_search import RAGSearcher
from opentelemetry import trace, context as otel_context
from opentelemetry.trace import SpanKind, StatusCode
from telemetry import (
    setup_telemetry, get_tracer,
    chat_request_counter, chat_error_counter, tool_called_counter,
    phase1_latency_histo,
)

load_dotenv()
setup_telemetry()

logger = logging.getLogger("tinywebapp.chat")
client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

TOOL_CONFIG = {
    "tools": [{
        "toolSpec": {
            "name": "rag_search",
            "description": (
                "Search the California benefits knowledge base. Use this when the user asks about "
                "California government benefits, programs, eligibility, or how to apply for assistance "
                "such as CalFresh, unemployment insurance, Medi-Cal, CalWORKs, WIC, SDI, PFL, "
                "CARE/FERA utilities discount, or job training programs."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query to find relevant benefit information"}
                    },
                    "required": ["query"],
                }
            },
        }
    }],
    "toolChoice": {"auto": {}},
}

SYSTEM_PROMPT = (
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


@app.get("/health")
def health():
    try:
        r.ping()
        redis_ok = True
    except Exception as exc:
        logger.warning("health_check.redis_fail", extra={"error": str(exc)})
        redis_ok = False
    return {"status": "ok" if redis_ok else "degraded", "redis": "ok" if redis_ok else "error"}


@app.post("/chat")
def chat(req: ChatRequest):
    tracer = get_tracer()
    session_id = req.session_id or str(uuid4())

    # Start span manually — it must outlive chat() since the generator runs later
    request_span = tracer.start_span(
        "chat",
        kind=SpanKind.SERVER,
        attributes={"http.method": "POST", "http.route": "/chat", "session.id": session_id},
    )
    ctx = trace.set_span_in_context(request_span)

    def stream():
        token = otel_context.attach(ctx)
        sources = []
        tool_called = False
        t_start = time.monotonic()

        try:
            raw = r.get(session_id)
            conversation_history = json.loads(raw) if raw else []
            conversation_history.append({'role': 'user', 'content': req.message})
            bedrock_messages = [{'role': m['role'], 'content': [{'text': m['content']}]} for m in conversation_history]

            # Phase 1: let Claude decide whether to search
            t0 = time.monotonic()
            with tracer.start_as_current_span("bedrock.converse", attributes={
                "gen_ai.system": "aws_bedrock",
                "gen_ai.request.model": settings.DEFAULT_MODEL.value,
                "gen_ai.request.max_tokens": 1024,
                "gen_ai.request.temperature": 0.2,
            }) as phase1_span:
                phase1 = client.converse(
                    modelId=settings.DEFAULT_MODEL.value,
                    messages=bedrock_messages,
                    system=[{'text': SYSTEM_PROMPT}],
                    toolConfig=TOOL_CONFIG,
                    inferenceConfig={'maxTokens': 1024, 'temperature': 0.2},
                )
                usage = phase1.get("usage", {})
                phase1_span.set_attribute("gen_ai.usage.input_tokens", usage.get("inputTokens", 0))
                phase1_span.set_attribute("gen_ai.usage.output_tokens", usage.get("outputTokens", 0))
                phase1_span.set_attribute("gen_ai.response.stop_reason", phase1["stopReason"])
            phase1_latency_histo.record(int((time.monotonic() - t0) * 1000))

            if phase1['stopReason'] == 'tool_use':
                tool_called = True
                tool_block = next(b['toolUse'] for b in phase1['output']['message']['content'] if 'toolUse' in b)
                query = tool_block['input']['query']
                tool_use_id = tool_block['toolUseId']

                request_span.set_attribute("rag.query", query)
                logger.info("tool_use", extra={"rag_query": query, "session_id": session_id})
                yield f"__status__:{query}\n"

                context = RAGSearcher().search(search_query=query)
                sources = [
                    {"citation_number": i+1, "title": d.get("title",""), "url": d.get("url",""), "excerpt": d.get("excerpt","")}
                    for i, d in enumerate(context)
                ]
                numbered_context = "\n\n".join(
                    f"[{i+1}] Title: {d['title']}\nURL: {d['url']}\n{d['excerpt']}"
                    for i, d in enumerate(context)
                )

                bedrock_messages = bedrock_messages + [
                    {'role': 'assistant', 'content': phase1['output']['message']['content']},
                    {'role': 'user', 'content': [{'toolResult': {'toolUseId': tool_use_id, 'content': [{'text': numbered_context}]}}]},
                ]

            request_span.set_attribute("tool.called", tool_called)

            full_text = []
            if phase1['stopReason'] != 'tool_use':
                text = phase1['output']['message']['content'][0]['text']
                full_text.append(text)
                yield text
            else:
                t_stream = time.monotonic()
                ttft_ms = None
                resp = client.converse_stream(
                    modelId=settings.DEFAULT_MODEL.value,
                    messages=bedrock_messages,
                    system=[{'text': SYSTEM_PROMPT}],
                    toolConfig=TOOL_CONFIG,
                    inferenceConfig={'maxTokens': 1024, 'temperature': 0.2},
                )
                for event in resp["stream"]:
                    if "contentBlockDelta" in event:
                        text = event["contentBlockDelta"]["delta"].get("text", "")
                        if text:
                            if ttft_ms is None:
                                ttft_ms = int((time.monotonic() - t_stream) * 1000)
                            full_text.append(text)
                            yield text
                request_span.set_attribute("gen_ai.time_to_first_token_ms", ttft_ms or 0)
                request_span.set_attribute("gen_ai.stream_duration_ms", int((time.monotonic() - t_stream) * 1000))

            final_text = "".join(full_text)
            conversation_history.append({'role': 'assistant', 'content': final_text})
            r.setex(session_id, 3600, json.dumps(conversation_history))

            chat_request_counter.add(1, {"tool_called": str(tool_called)})
            if tool_called:
                tool_called_counter.add(1)

            logger.info("chat_complete", extra={
                "session_id": session_id,
                "tool_called": tool_called,
                "total_duration_ms": int((time.monotonic() - t_start) * 1000),
            })
            request_span.set_status(StatusCode.OK)

        except Exception as e:
            request_span.record_exception(e)
            request_span.set_status(StatusCode.ERROR, str(e))
            chat_error_counter.add(1)
            logger.error("chat_error", extra={"error": str(e), "session_id": session_id}, exc_info=True)
            yield "Sorry, something went wrong. Please try again."

        finally:
            yield f"\n__sources__:{json.dumps(sources)}"
            request_span.end()
            otel_context.detach(token)

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )
