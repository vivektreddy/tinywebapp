import os, json
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

load_dotenv()

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


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid4())

    def stream():
        sources = []
        try:
            raw = r.get(session_id)
            conversation_history = json.loads(raw) if raw else []
            conversation_history.append({'role': 'user', 'content': req.message})
            bedrock_messages = [{'role': m['role'], 'content': [{'text': m['content']}]} for m in conversation_history]

            # Phase 1: let Claude decide whether to search
            phase1 = client.converse(
                modelId=settings.DEFAULT_MODEL.value,
                messages=bedrock_messages,
                system=[{'text': SYSTEM_PROMPT}],
                toolConfig=TOOL_CONFIG,
                inferenceConfig={'maxTokens': 1024, 'temperature': 0.2},
            )

            if phase1['stopReason'] == 'tool_use':
                tool_block = next(b['toolUse'] for b in phase1['output']['message']['content'] if 'toolUse' in b)
                query = tool_block['input']['query']
                tool_use_id = tool_block['toolUseId']
                print(f"[tool_use] rag_search query: {query}")

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

            full_text = []
            if phase1['stopReason'] != 'tool_use':
                text = phase1['output']['message']['content'][0]['text']
                full_text.append(text)
                yield text
            else:
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
                            full_text.append(text)
                            yield text

            final_text = "".join(full_text)
            conversation_history.append({'role': 'assistant', 'content': final_text})
            r.setex(session_id, 3600, json.dumps(conversation_history))

        except Exception as e:
            print(f"[error] stream(): {e}")
            yield "Sorry, something went wrong. Please try again."

        finally:
            yield f"\n__sources__:{json.dumps(sources)}"

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )
