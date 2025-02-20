import sounddevice as sd
import json
import scipy.io.wavfile as wav
import numpy as np
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from google import genai
import re

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

def save_audio(audio_data, sample_rate, filename):
    """Save audio data to a WAV file."""
    wav.write(filename, sample_rate, audio_data)
    return filename

def transcribe_audio(filepath):
    """Transcribe audio file using Deepgram."""
    try:
        deepgram = DeepgramClient(api_key='a8b75fa07ad77e26a7866d995ed329553927767b')
        with open(filepath, 'rb') as file:
            buffer_data = file.read()
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-3",
            # filler_words=True,
            language='en',
            numerals=True,
        )
        print("Transcribing audio...")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        return response
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def parse_transcript(response):
    """Parse the Deepgram response to extract only the relevant information."""
    try:
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        words = response["results"]["channels"][0]["alternatives"][0]["words"]
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

def display_transcript(transcript_json):
    """Display the transcript JSON in a formatted way."""
    if transcript_json:
        print(json.dumps(transcript_json, indent=2))

def optimize_transcript_with_gemini(transcription_json, gemini_api_key):
    """
    Optimize the transcription JSON by removing duplicate phrases.
    
    The prompt instructs Gemini to:
      - Remove any duplicate phrases.
      - If a phrase is repeated, keep only the last occurrence (with its timings).
      - Return the result in the same JSON schema:
      
        {
          "transcript": str,
          "words": [
            {
              "word": str,
              "timing": {
                 "start": float,
                 "end": float
              }
            },
            ...
          ]
        }
    """
    client = genai.Client(api_key=gemini_api_key)
    prompt = (
        "Optimize the following transcription JSON by removing duplicate phrases. "
        "If there are repeated segments, remove the duplicates and keep only the last occurrence, "
        "preserving the start and end times of the retained words. "
        "Return the result in the exact JSON format as shown below:\n\n"
        "{\n"
        "  \"transcript\": str,\n"
        "  \"words\": [\n"
        "    {\n"
        "      \"word\": str,\n"
        "      \"timing\": {\n"
        "         \"start\": float,\n"
        "         \"end\": float\n"
        "      }\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Input: " + json.dumps(transcription_json)
    )

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
    )
    cleaned_text = re.sub(r'^```json\n|```$', '', response.text.strip())

    try:
        optimized_json = json.loads(cleaned_text)
        return optimized_json
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return transcription_json

# ----- Updated enhance_audio Function with Volume Normalization and Fade In/Out Transitions -----
def enhance_audio(original_audio, sample_rate, optimized_transcript):
    """
    Create an enhanced audio file by extracting and concatenating only the segments
    corresponding to the optimized transcript's words, while applying:
      - Volume Normalization: Normalize each segment's volume to a target level.
      - Fade In/Out Transitions: Apply short fade in/out effects at segment boundaries.
    """
    segments = []
    fade_duration_ms = 50  # 50 milliseconds fade duration
    fade_duration_samples = int(sample_rate * fade_duration_ms / 1000)
    target_amplitude = 30000  # Target max amplitude for normalization

    for word_info in optimized_transcript.get("words", []):
        timing = word_info.get("timing", {})
        start = timing.get("start", 0)
        end = timing.get("end", 0)
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment = original_audio[start_idx:end_idx]
        if segment.size == 0:
            continue

        # ----- Volume Normalization -----
        current_max = np.max(np.abs(segment))
        if current_max > 0:
            normalization_factor = target_amplitude / current_max
        else:
            normalization_factor = 1.0
        normalized_segment = np.int16(segment * normalization_factor)

        # ----- Fade In/Out Transitions -----
        seg_length = normalized_segment.shape[0]
        fade_samples = min(fade_duration_samples, seg_length // 2)
        # Create fade-in envelope (linear ramp)
        fade_in = np.linspace(0, 1, fade_samples, endpoint=True)
        # Create fade-out envelope (linear ramp)
        fade_out = np.linspace(1, 0, fade_samples, endpoint=True)
        # Apply fade in to the beginning
        normalized_segment[:fade_samples, 0] = (normalized_segment[:fade_samples, 0] * fade_in).astype(np.int16)
        # Apply fade out to the end
        normalized_segment[-fade_samples:, 0] = (normalized_segment[-fade_samples:, 0] * fade_out).astype(np.int16)
        
        segments.append(normalized_segment)
    
    if segments:
        enhanced_audio = np.concatenate(segments, axis=0)
    else:
        enhanced_audio = original_audio
    return enhanced_audio

def main():
    try:
        # Get recording duration from the user.
        duration = float(input("Enter recording duration in seconds: "))
        
        # Record audio.
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save the original raw audio.
        raw_audio_file = save_audio(audio_data, sample_rate, "recorded_audio.wav")
        print(f"Raw audio saved to: {raw_audio_file}")
        
        # Transcribe the raw audio.
        transcript = transcribe_audio(raw_audio_file)
        if transcript is None:
            print("Transcription failed.")
            return
        
        # Parse and display the raw transcript.
        parsed_output = parse_transcript(transcript)
        print("Raw Transcript:")
        display_transcript(parsed_output)
        
        # Optimize the transcript using the Gemini LLM.
        gemini_api_key = "AIzaSyAOK9vRTSRQzd22B2gmbiuIePbZTDyaGYs"  # Replace with your actual Gemini API key.
        optimized_output = optimize_transcript_with_gemini(parsed_output, gemini_api_key)
        print("\nOptimized Transcript:")
        display_transcript(optimized_output)
        
        # Enhance the audio by extracting segments from the optimized transcript with normalization and fades.
        enhanced_audio = enhance_audio(audio_data, sample_rate, optimized_output)
        enhanced_audio_file = save_audio(enhanced_audio, sample_rate, "enhanced_audio.wav")
        print(f"\nEnhanced audio saved to: {enhanced_audio_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
