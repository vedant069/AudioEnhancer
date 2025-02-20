import os
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from groq import Groq

# Initialize the Groq client
client = Groq(api_key="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")


def transcribe_audio(filepath):
    """
    Transcribe audio file using Groq's Whisper implementation with pronunciation details
    """
    try:
        with open(filepath, "rb") as file:
            # Get verbose JSON format with word-level details
            verbose_response = client.audio.transcriptions.create(
                # filepath="recorded_audio2.wav",
                file=(filepath, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                language="en",
                # temperature=0.0,
                # word_timestamps=True  # Enable word-level timing
            )
            print(verbose_response)
        return verbose_response
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return None
x = transcribe_audio("recorded_audio2.wav")
print(x)