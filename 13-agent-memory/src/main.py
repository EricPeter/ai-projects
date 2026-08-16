from fastapi import FastAPI
from pydantic import BaseModel
from .agent import chat
from .memory import short_term, long_term

app = FastAPI(title="Agent Memory System")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default-user"


@app.post("/sessions/{session_id}/chat")
def session_chat(session_id: str, body: ChatRequest):
    answer = chat(session_id, body.user_id, body.message)
    return {"answer": answer, "session_id": session_id}


@app.get("/sessions/{session_id}/memory")
def get_session_memory(session_id: str):
    return {"turns": short_term.get_turns(session_id)}


@app.delete("/sessions/{session_id}/memory")
def clear_session_memory(session_id: str):
    short_term.clear(session_id)
    return {"cleared": True}


@app.get("/users/{user_id}/memories")
def get_user_memories(user_id: str):
    return {"memories": long_term.get_all(user_id)}


@app.get("/health")
def health():
    return {"status": "ok"}
