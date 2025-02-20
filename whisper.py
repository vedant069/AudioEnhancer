import os
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from groq import Groq

# Initialize the Groq client
client = Groq(api_key="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")

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

def transcribe_audio(filepath):
    """
    Transcribe audio file using Groq's Whisper implementation with pronunciation details
    """
    try:
        with open(filepath, "rb") as file:
            # Get verbose JSON format with word-level details
            verbose_response = client.audio.transcriptions.create(
                filepath="recorded_audio2.wav",
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

def display_pronunciation_details(response):
    """
    Display pronunciation details from the verbose response
    """
    if not response:
        return

    print("\nTranscription with Timing Details:")
    try:
        segments = response.segments
        for segment in segments:
            print(f"\nSegment text: {segment.text}")
            print(f"Start time: {segment.start:.2f}s")
            print(f"End time: {segment.end:.2f}s")
            print(f"Confidence: {segment.confidence:.2%}")
            
            if hasattr(segment, 'words'):
                print("Word-by-word breakdown:")
                for word in segment.words:
                    print(f"  '{word.word}' ({word.start:.2f}s - {word.end:.2f}s)")

    except AttributeError as e:
        print(f"Could not parse pronunciation details: {e}")

def main():
    try:
        # Get recording duration from user
        duration = float(input("Enter recording duration in seconds: "))
        
        # Record audio
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save audio to file
        # audio_file = save_audio(audio_data, sample_rate)
        audio_file = "harvard.wav"
        print(f"Audio saved to: {audio_file}")
        
        # Transcribe the audio
        response = transcribe_audio(audio_file)
        
        if response:
            # Display pronunciation details
            display_pronunciation_details(response)
            
            # Optionally, clean up the audio file
            os.remove(audio_file)
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()

