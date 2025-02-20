import os
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import re
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)


def transcribe_audio(filepath, api_key):
    """
    Transcribe audio file using Deepgram
    
    Args:
        filepath (str): Path to the audio file
        api_key (str): Deepgram API key
    
    Returns:
        dict: Transcription response
    """
    try:
        deepgram = DeepgramClient(api_key=api_key)
        
        with open(filepath, 'rb') as file:
            buffer_data = file.read()
        
        payload: FileSource = {
            "buffer": buffer_data,
        }
        
        options = PrerecordedOptions(
            # model="whisper-large",
            model="nova-3",
            # utt_split=1,
            utterances=True,
            
            numerals=True,
            punctuate=True,
            paragraphs=True,
            filler_words=True,
            

        )
        
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        return response
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

x = transcribe_audio("recorded_audio2.wav", "a8b75fa07ad77e26a7866d995ed329553927767b")
print(x)


