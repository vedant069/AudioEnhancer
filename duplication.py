import os
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from google import genai
import re


##############################
# 1. AUDIO RECORDING & SAVING
##############################
def record_audio(duration=5, sample_rate=44100):
    """
    Record audio from the microphone.
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
    Save recorded audio to a WAV file.
    """
    wav.write(filename, sample_rate, audio_data)
    return filename

####################################
# 2. AUDIO TRANSCRIPTION (Deepgram)
####################################
def transcribe_audio(filepath, deepgram_api_key):
    """
    Transcribe an audio file using Deepgram Nova.
    """
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
            utt_split=1,
            utterances=True,
            language='en',
            numerals=True,
            punctuate=True,
        )
        print("Transcribing audio...")
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        
        # Debug print to see response structure
        print("Deepgram response structure:")
        print(json.dumps(response.to_dict(), indent=2))
        
        return response
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

############################################################
# 3. REDUCE TRANSCRIPTION JSON TO KEEP ONLY IMPORTANT DATA
############################################################
def reduce_transcription_json(transcription):
    """
    Reduce the transcription JSON by keeping only the necessary fields.
    """
    # Convert Deepgram response to dictionary if needed
    if hasattr(transcription, 'to_dict'):
        transcription = transcription.to_dict()

    reduced = {"utterances": []}
    
    # Navigate the correct path in Deepgram's response structure
    channels = transcription.get("results", {}).get("channels", [])
    if not channels:
        return reduced
        
    # Get the first channel's alternatives
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return reduced
        
    # Get words from the first alternative
    words = alternatives[0].get("words", [])
    
    # Group words into utterances based on timing and punctuation
    current_utterance = None
    
    for word in words:
        if current_utterance is None:
            current_utterance = {
                "start": word.get("start"),
                "end": word.get("end"),
                "transcript": word.get("word"),
                "words": [word]
            }
        else:
            # If there's a long pause or punctuation, start a new utterance
            if word.get("start") - current_utterance["end"] > 1.0 or \
               word.get("punctuated_word", "").endswith((".", "!", "?")):
                reduced["utterances"].append(current_utterance)
                current_utterance = {
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "transcript": word.get("word"),
                    "words": [word]
                }
            else:
                current_utterance["end"] = word.get("end")
                current_utterance["transcript"] += " " + word.get("word")
                current_utterance["words"].append(word)
    
    # Add the last utterance if exists
    if current_utterance is not None:
        reduced["utterances"].append(current_utterance)
    
    return reduced

############################################################
# 4. FILTER TRANSCRIPTION WITH GEMINI-2.0-FLASH (Google GenAI)
############################################################
def filter_transcription_with_gemini(reduced_json, gemini_api_key):
    """
    Pass the reduced transcription JSON to Gemini-2.0-Flash to identify the
    important responses. The Gemini prompt instructs the model to:
      - Remove duplicate or erroneous self-introduction utterances (keeping only the final one)
      - Return a JSON object with:
            'filtered_transcription': the updated transcription with only the important utterances,
            'removal_segments': a list of [start, end] time segments (in seconds) corresponding to parts that should be removed.
    
    The timestamps should be kept as provided.
    """
    # Build the prompt with instructions and the reduced JSON input.
    prompt = f"""You are an audio transcription editor for voice-over recordings. You are given a JSON object representing a transcription. The JSON object has a key "utterances", which is a list of utterance objects. Each utterance object contains:
- "start": the start time (in seconds),
- "end": the end time (in seconds),
- "transcript": the full text of the utterance,
- "words": a list of word objects (each having "word", "start", and "end").
    
The speaker has accidentally recorded duplicate audio. For example, if there are multiple utterances like "Hi. My name is Vedant.", only the final self-introduction is correct. Remove any previous self-introduction or any other duplications utterances and keep only the final correct one, preserving the timestamps. Also, remove any extraneous fields so that only the mentioned fields remain.Make sure their are no duplications in the transcription.
    
Return a JSON object with two keys:
1. "filtered_transcription": the updated transcription JSON (with key "utterances") containing only the necessary utterances.
2. "removal_segments": a list of [start, end] time segments (in seconds) corresponding to the parts of the audio that should be removed.
    
Do not include any additional commentary. Use the input JSON exactly as provided.
    
Input JSON:
{json.dumps(reduced_json, indent=2)}
"""
    try:
        client = genai.Client(api_key='AIzaSyAOK9vRTSRQzd22B2gmbiuIePbZTDyaGYs')
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        cleaned_text = re.sub(r'^```json\n|```$', '', response.text.strip())        # The response text is expected to be a JSON string.
        result = json.loads(cleaned_text)
        print(response)
        print(result)
        return result
    except Exception as e:
        print(f"Error processing transcription with Gemini: {e}")
        return None

############################################################
# 5. ENHANCE AUDIO BY REMOVING UNWANTED SEGMENTS
############################################################
def enhance_audio_by_segments(audio_filepath, removal_segments, output_filename="enhanced_audio.wav"):
    """
    Remove segments from the audio file corresponding to the time ranges provided
    in removal_segments and save the enhanced audio.
    
    removal_segments: list of [start, end] segments (in seconds) to remove.
    """
    try:
        sample_rate, data = wav.read(audio_filepath)
        # If the audio has more than one channel, take only the first channel.
        if data.ndim > 1:
            data = data[:, 0]
        
        # Convert each time segment from seconds to sample indices.
        segments_to_remove = [
            (int(seg[0] * sample_rate), int(seg[1] * sample_rate))
            for seg in removal_segments
        ]
        segments_to_remove.sort(key=lambda x: x[0])
        
        # Merge overlapping or contiguous segments.
        merged_segments = []
        for seg in segments_to_remove:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_seg = merged_segments[-1]
                if seg[0] <= last_seg[1]:
                    merged_segments[-1] = (last_seg[0], max(last_seg[1], seg[1]))
                else:
                    merged_segments.append(seg)
        
        # Build the list of audio segments to keep.
        keep_segments = []
        current_idx = 0
        for start_idx, end_idx in merged_segments:
            if start_idx > current_idx:
                keep_segments.append(data[current_idx:start_idx])
            current_idx = end_idx
        if current_idx < len(data):
            keep_segments.append(data[current_idx:])
        
        if keep_segments:
            enhanced_audio = np.concatenate(keep_segments)
        else:
            enhanced_audio = data
        
        wav.write(output_filename, sample_rate, enhanced_audio)
        print(f"Enhanced audio saved to {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error during audio enhancement: {e}")
        return None

####################################
# 6. MAIN PIPELINE
####################################
def main():
    try:
        # Replace with your actual API keys.
        deepgram_api_key = "a8b75fa07ad77e26a7866d995ed329553927767b"
        gemini_api_key = "AIzaSyAOK9vRTSRQzd22B2gmbiuIePbZTDyaGYs"
        
        duration = float(input("Enter recording duration in seconds: "))
        
        # 1. Record and save audio.
        audio_data, sample_rate = record_audio(duration=duration)
        audio_file = save_audio(audio_data, sample_rate)
        print(f"Audio saved to: {audio_file}")
        
        # 2. Transcribe the audio using Deepgram.
        transcription = transcribe_audio(audio_file, deepgram_api_key)
        if transcription is None:
            print("Transcription failed. Exiting.")
            return
        
        # 3. Reduce the transcription JSON.
        reduced_json = reduce_transcription_json(transcription)
        # (Optional) Print the reduced JSON:
        # print(json.dumps(reduced_json, indent=2))
        
        # 4. Pass the reduced transcription to Gemini-2.0-Flash for filtering.
        gemini_result = filter_transcription_with_gemini(reduced_json, gemini_api_key)
        if gemini_result is None:
            print("Gemini filtering failed. Exiting.")
            return
        
        # Expect gemini_result to have keys: 'filtered_transcription' and 'removal_segments'
        filtered_transcription = gemini_result.get("filtered_transcription")
        removal_segments = gemini_result.get("removal_segments", [])
        print("Filtered transcription (from Gemini):")
        print(json.dumps(filtered_transcription, indent=2))
        print("Removal segments (from Gemini):")
        print(removal_segments)
        
        # 5. Enhance the audio by removing the unwanted segments.
        enhanced_file = enhance_audio_by_segments(audio_file, removal_segments, output_filename="enhanced_audio.wav")
        
        # Optionally, remove the original audio file.
        # os.remove(audio_file)
        
    except Exception as e:
        print(f"Exception in main: {e}")

if __name__ == "__main__":
    main()