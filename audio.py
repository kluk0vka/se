from transformers import pipeline
import soundfile as sf

text = """
Welcome to Fight Club.
The first rule of Fight Club is: you do not talk about Fight Club.
The second rule of Fight Club is: you DO NOT talk about Fight Club.
"""

narrator = pipeline(
    "text-to-speech",
    model="suno/bark-small"
)

narrated_text = narrator(text)

sf.write(
    "narrated_text.wav",
    narrated_text["audio"][0],
    narrated_text["sampling_rate"]
)

print("Audio file saved as narrated_text.wav")
