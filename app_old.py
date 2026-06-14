import os
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

#client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI()

conversation_history : List[Dict[str,str]] = []


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    conversation_history.append({'role':'user', 'content':req.message})
    #response = client.responses.create(model='gpt-5.5',input=conversation_history)
    response = client.messages.create(model='claude-opus-4-8',max_tokens=1024,messages = conversation_history)
    #conversation_history.append({'role':'assistant', 'content':response.output_text})


load_dotenv()

#client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI()

conversation_history : List[Dict[str,str]] = []


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    conversation_history.append({'role':'user', 'content':req.message})
    #response = client.responses.create(model='gpt-5.5',input=conversation_history)
    response = client.messages.create(model='claude-opus-4-8',max_tokens=1024,messages = conversation_history)
    #conversation_history.append({'role':'assistant', 'content':response.output_text})
    conversation_history.append({'role':'assistant', 'content':response.content[0].text})
    #return {"response": response.output_text, 'history': conversation_history}
    return {"response": response.content[0].text, 'history': conversation_history}
