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

def record_audio(duration=5, sample_rate=44100):
    """
    Record audio from the microphone
    
    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Sample rate for audio recording
    
    Returns:
        numpy.ndarray: Recorded audio data
        int: Sample rate
    """
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")
    return audio_data, sample_rate

def save_audio(audio_data, sample_rate, filename="recorded_audio.wav"):
    """
    Save recorded audio to a WAV file
    
    Args:
        audio_data (numpy.ndarray): Audio data to save
        sample_rate (int): Sample rate of the audio
        filename (str): Output filename
    """
    wav.write(filename, sample_rate, audio_data)
    return filename



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
            model="nova-3",
            filler_words=True,
            utt_split=1,
            utterances=True,
            language='en',
            numerals=True,
            punctuate=True,
        )
        print("Transcribing audio...")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        return response
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def main():
    try:
        api_key = "a8b75fa07ad77e26a7866d995ed329553927767b"  
        
        duration = float(input("Enter recording duration in seconds: "))
        
        audio_data, sample_rate = record_audio(duration=duration)
        audio_file = save_audio(audio_data, sample_rate)
        print(f"Audio saved to: {audio_file}")
        
        response = transcribe_audio(audio_file, api_key)
        print(response)
            
        os.remove(audio_file)
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()