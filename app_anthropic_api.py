import os, json
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from anthropic import Anthropic
from schemas import Source, ChatRequest, ChatResponse
from uuid import uuid4

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI()

sessions = {}

@app.post("/chat",response_model = ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid4())
    conversation_history = sessions.get(session_id, [])
    conversation_history.append({'role':'user', 'content':req.message})
    response = client.messages.create(model='claude-opus-4-8',max_tokens=1024,messages = conversation_history)
    conversation_history.append({'role':'assistant', 'content':response.content[0].text})
    sessions[session_id] = conversation_history
    return ChatResponse(response= response.content[0].text, session_id = session_id, sources = [])
