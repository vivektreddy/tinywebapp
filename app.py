import os, json
import redis
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
import boto3
from schemas import Source, ChatRequest, ChatResponse
from uuid import uuid4
from config import settings
from rag_search import RAGSearcher

load_dotenv()

client = boto3.client("bedrock-runtime", region_name = settings.AWS_REGION)

app = FastAPI()

#sessions = {}
r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

@app.post("/chat",response_model = ChatResponse)
def chat(req: ChatRequest):
    

    context = RAGSearcher().search(search_query = req.message)
    print('context: ',context)

    SYSTEM_PROMPT = "You are a helpful concise assistant to help people in California " + \
    "who just lost their jobs.  Reply with a very pompous and sophisticated tone. " + \
    "Base your answers on the retrieved documents and cite your sources in line in the answer. " + \
    "If no documents pertain to the user's question, use your general knowledge and say so" + \
    "Do not answer questions unrelated to helping user. " +  \
    f"Here's the retrieved documents to help answer your question: {context}"


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
    response = client.converse(modelId = settings.DEFAULT_MODEL.value, \
    messages = bedrock_messages, \
    system = [{'text':SYSTEM_PROMPT}],inferenceConfig = {'maxTokens':1024,'temperature':0.2})
    output_text =  response["output"]["message"]["content"][0]["text"]
    conversation_history.append({'role':'assistant', 'content':output_text})
    #update session history
    #sessions[session_id] = conversation_history
    r.setex(session_id, 3600, json.dumps(conversation_history)) 
    sources = [Source(title=s.get("title", ""), url=s.get("url"), excerpt=s.get("excerpt", "")) for s in context]
    return ChatResponse(response=output_text, session_id=session_id, sources=sources)
