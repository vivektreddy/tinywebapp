import os, json
import redis
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import boto3
from schemas import Source, ChatRequest, ChatResponse
from uuid import uuid4
from config import settings
from rag_search import RAGSearcher

load_dotenv()

client = boto3.client("bedrock-runtime", region_name = settings.AWS_REGION)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id", "X-Sources"],
)

#sessions = {}
r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

@app.post("/chat")
async def chat(req: ChatRequest):

    context = RAGSearcher().search(search_query = req.message)
    print('context: ',context)

    numbered_context = "\n\n".join(
        f"[{i+1}] Title: {doc.get('title','')}\nURL: {doc.get('url','')}\n{doc.get('excerpt','')}"
        for i, doc in enumerate(context)
    )
    SYSTEM_PROMPT = "You are a helpful concise assistant to help people in California " + \
    "who just lost their jobs.  Reply with a very pompous and condescending tone. " + \
    "Base your answers on the retrieved documents. Cite sources inline using bracketed numbers like [1] or [2]. " + \
    "If no documents pertain to the user's question, use your general knowledge and say so. " + \
    "Do not answer questions unrelated to helping user. " + \
    f"Here are the retrieved documents:\n\n{numbered_context}"


    #create or retrieve session id
    session_id = req.session_id or str(uuid4())
    #conversation_history = sessions.get(session_id, [])
    raw = r.get(session_id)
    conversation_history = json.loads(raw) if raw else []
    #add sessions history
    conversation_history.append({'role':'user', 'content':req.message})
    #put in proper bedrock message format
    bedrock_messages = [{'role':msg['role'], 'content': [{'text': msg['content']}]} for msg in conversation_history]
    #call bedrock model and append result


    def stream():
        full_text = []
        response = client.converse_stream(modelId = settings.DEFAULT_MODEL.value, \
        messages = bedrock_messages, \
        system = [{'text':SYSTEM_PROMPT}],\
        inferenceConfig = {'maxTokens':1024,'temperature':0.2})

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                text = delta.get("text","")
                if text:
                    full_text.append(text)
                    yield text
        final_text = "".join(full_text)
        conversation_history.append({'role':'assistant', 'content':final_text})
        r.setex(session_id, 3600, json.dumps(conversation_history)) 
    sources = [
        {"citation_number": i+1, "title": d.get("title",""), "url": d.get("url",""), "excerpt": d.get("excerpt","")}
        for i, d in enumerate(context)
    ]
    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={
            "X-Session-Id": session_id,
            "X-Sources": json.dumps(sources),
        },
    )
