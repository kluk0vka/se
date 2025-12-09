from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # новое
from pydantic import BaseModel
from backend.agent import agent

app = FastAPI(title="AI-Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

@app.post("/process")
def process(data: TextIn):
    try:
        return {"result": agent.run(data.text)}
    except Exception as e:
        return {"result": str(e).split("Final Answer:")[-1].strip() or "Ответ получен."}

@app.get("/")
def root():
    return {"message": "POST /process  text:string"}