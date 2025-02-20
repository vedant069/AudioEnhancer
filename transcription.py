import sounddevice as sd
import json
import scipy.io.wavfile as wav
import numpy as np
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

def record_audio(duration=5, sample_rate=44100):
    """Record audio from the microphone."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype=np.int16)
    sd.wait()
    print("Recording finished!")
    return audio_data, sample_rate

def save_audio(audio_data, sample_rate, filename="recorded_audio.wav"):
    """Save recorded audio to a WAV file."""
    wav.write(filename, sample_rate, audio_data)
    return filename

def transcribe_audio(filepath, api_key):
    """Transcribe audio file using Deepgram."""
    try:
        deepgram = DeepgramClient(api_key='a8b75fa07ad77e26a7866d995ed329553927767b')
        
        with open(filepath, 'rb') as file:
            buffer_data = file.read()
        
        payload: FileSource = {
            "buffer": buffer_data,
        }
        
        options = PrerecordedOptions(
            model="nova-3",
            filler_words=True,
            # utt_split=1,
            # utterances=True,
            language='en',
            numerals=True,
            # punctuate=True,
        )
        print("Transcribing audio...")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        return response
    except Exception as e:
        print(f"Transcription error: {e}")

        return None

def parse_transcript(response):
    """Parse the Deepgram response to extract only relevant information."""
    try:
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        words = response["results"]["channels"][0]["alternatives"][0]["words"]
        
        # Create a clean JSON structure
        parsed_output = {
            "transcript": transcript,
            "words": [
                {
                    "word": word["word"],
                    "timing": {
                        "start": round(word["start"], 2),
                        "end": round(word["end"], 2)
                    }
                }
                for word in words
            ]
        }
        return parsed_output
    except Exception as e:
        print(f"Error parsing transcript: {e}")
        return None

def display_transcript(parsed_output):
    """Display the transcript in JSON format."""
    if not parsed_output:
        return
    
    print(json.dumps(parsed_output, indent=2))

def main():
    try:
        # Get recording duration from user
        duration = float(input("Enter recording duration in seconds: "))
        
        # Record and save audio
        audio_data, sample_rate = record_audio(duration=duration)
        audio_file = save_audio(audio_data, sample_rate)
        print(f"Audio saved to: {audio_file}")
        
        # Transcribe the audio
        transcript = transcribe_audio(audio_file, None)  # api_key is hardcoded in transcribe_audio
        if transcript is None:
            print("Transcription failed.")
            return
        
        # Parse and display the transcript
        parsed_output = parse_transcript(transcript)
        display_transcript(parsed_output)
        
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()