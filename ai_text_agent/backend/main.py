from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.agent import agent
from langfuse import Langfuse, observe


langfuse = Langfuse(
    public_key="pk-lf-f87c1a59-549d-40a1-bfa6-57d5fab01530",
    secret_key="sk-lf-9adc28ea-a641-4b11-9e4c-81165ac06023",
    host="http://localhost:3000",
)

app = FastAPI(title="AI-Agent with Observability")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str


@observe(name="agent_process")
def run_agent_with_trace(text: str) -> str:
    return agent.run(text)


@app.post("/process")
def process(data: TextIn):
    try:
        result = run_agent_with_trace(data.text)
        return {"result": result}
    except Exception as e:
        return {"result": str(e) or "Ответ получен."}


@app.get("/")
def root():
    return {"message": "POST /process { text: string }"}

