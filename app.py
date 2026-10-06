from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import answer_question


app = FastAPI(
    title="Research Paper RAG Assistant",
    description="Ask questions about research papers using RAG.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def root():

    return {
        "message": "Research Paper RAG Assistant is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = answer_question(
        request.question
    )

    return result