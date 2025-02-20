import sounddevice as sd
import json
import scipy.io.wavfile as wav
import numpy as np
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from google import genai
import re
import noisereduce as nr

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

def purify_audio(audio_data, sample_rate):
    """
    Purify the audio by removing background noise.
    
    This function converts the int16 audio data to normalized float32,
    uses the first 0.5 seconds as a noise profile, applies noise reduction,
    and then converts the result back to int16.
    """
    # Convert to float32 in range [-1, 1]
    audio_float = audio_data.astype(np.float32) / 32768.0
    # Assume the first 0.5 seconds is mainly background noise
    noise_duration = 0.5
    noise_sample = audio_float[:int(noise_duration * sample_rate)]
    
    # Flatten in case audio is 2D (mono with shape [samples, 1])
    audio_flat = audio_float.flatten()
    noise_flat = noise_sample.flatten()
    
    # Use the correct keyword arguments for noisereduce.reduce_noise()
    purified_float = nr.reduce_noise(y=audio_flat, sr=sample_rate, y_noise=noise_flat)






    
    # Convert back to int16 (scaling and clipping)
    purified_int16 = np.int16(np.clip(purified_float, -1.0, 1.0) * 32767)
    # Reshape to original shape (samples, 1)
    purified_int16 = purified_int16.reshape(-1, 1)
    return purified_int16

def transcribe_audio(filepath):
    """Transcribe audio file using Deepgram."""
    try:
        # The Deepgram API key is hardcoded here.
        deepgram = DeepgramClient(api_key='a8b75fa07ad77e26a7866d995ed329553927767b')
        with open(filepath, 'rb') as file:
            buffer_data = file.read()
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-3",
            filler_words=True,
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
    # Remove any markdown formatting from the response.
    cleaned_text = re.sub(r'^```json\n|```$', '', response.text.strip())

    try:
        optimized_json = json.loads(cleaned_text)
        return optimized_json
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return transcription_json

def enhance_audio(original_audio, sample_rate, optimized_transcript):
    """
    Create an enhanced audio file by extracting and concatenating only the segments
    corresponding to the optimized transcript's words.
    """
    segments = []
    for word_info in optimized_transcript.get("words", []):
        timing = word_info.get("timing", {})
        start = timing.get("start", 0)
        end = timing.get("end", 0)
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        # Append the segment corresponding to this word.
        segments.append(original_audio[start_idx:end_idx])
    
    if segments:
        # Concatenate all segments along the first axis.
        enhanced_audio = np.concatenate(segments, axis=0)
    else:
        enhanced_audio = original_audio
    return enhanced_audio

def main():
    try:
        # Get recording duration from the user.
        duration = float(input("Enter recording duration in seconds: "))
        
        # Record audio.
        raw_audio_data, sample_rate = record_audio(duration=duration)
        
        # Save the original raw audio.
        raw_audio_file = save_audio(raw_audio_data, sample_rate, "recorded_audio.wav")
        print(f"Raw audio saved to: {raw_audio_file}")
        
        # Purify the raw audio by removing background noise.
        purified_audio_data = purify_audio(raw_audio_data, sample_rate)
        purified_audio_file = save_audio(purified_audio_data, sample_rate, "purified_audio.wav")
        print(f"Purified audio saved to: {purified_audio_file}")
        
        # Transcribe the purified audio.
        transcript = transcribe_audio(purified_audio_file)
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
        
        # Enhance the audio by extracting segments from the optimized transcript.
        enhanced_audio = enhance_audio(purified_audio_data, sample_rate, optimized_output)
        enhanced_audio_file = save_audio(enhanced_audio, sample_rate, "enhanced_audio.wav")
        print(f"\nEnhanced audio saved to: {enhanced_audio_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
