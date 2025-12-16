from transformers import pipeline
import torch
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
sentiment_pipe = pipeline(
    "text-classification",
    model="blanchefort/rubert-base-cased-sentiment",
    device=0 if device == "mps" else -1
)

def get_sentiment(text: str) -> str:
    return sentiment_pipe(text)[0]["label"]