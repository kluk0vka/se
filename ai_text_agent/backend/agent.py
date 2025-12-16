from langchain.agents import initialize_agent, AgentType
from langchain.llms.base import LLM
from backend.tools import summarize, translate, sentiment
import ollama

class OllamaLLM(LLM):
    model: str = "llama3.2:3b"
    def _call(self, prompt: str, stop=None):
        resp = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]
    @property
    def _llm_type(self): return "ollama"

agent = initialize_agent(
    tools=[summarize, translate, sentiment],
    llm=OllamaLLM(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=10,
    early_stopping_method="generate"
)