# api/main.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import json

from agent.agent import SHLAgent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role: str
    content: str
    @validator('role')
    def check_role(cls, v):
        if v not in ('user', 'assistant'):
            raise ValueError('role must be user or assistant')
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False

agent = None

def get_agent():
    global agent
    if agent is None:
        agent = SHLAgent()
    return agent

@app.on_event("startup")
async def startup():
    get_agent()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(messages: List[Message]):
    try:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        result = get_agent().process(msg_dicts)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(
            reply="I encountered an issue. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False
        )