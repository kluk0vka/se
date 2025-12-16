from langchain.tools import tool
from backend.model import get_sentiment
import ollama

@tool
def summarize(text: str) -> str:
    """Суммаризировать длинный текст."""
    resp = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": f"Кратко: {text}"}])
    return resp["message"]["content"]

@tool
def translate(text: str) -> str:
    """Перевести русский текст на английский."""
    resp = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": f"Translate: {text}"}])
    return resp["message"]["content"]

@tool
def sentiment(text: str) -> str:
    """Определить тональность текста."""
    return get_sentiment(text)