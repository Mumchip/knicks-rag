"""
api.py — FastAPI server.
Run: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import rag

app = FastAPI(title="Knicks RAG")
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class ChatRequest(BaseModel):
    question: str


class ArgueRequest(BaseModel):
    take: str


class ChatResponse(BaseModel):
    answer: str


class ArgueResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    try:
        count = rag._collection.count()
        return {"status": "ok", "documents": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/")
def root():
    return FileResponse("frontend/index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return ChatResponse(answer=rag.answer(req.question))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/argue", response_model=ArgueResponse)
def argue(req: ArgueRequest):
    if not req.take.strip():
        raise HTTPException(status_code=400, detail="Take cannot be empty")
    return ArgueResponse(response=rag.argue(req.take))
