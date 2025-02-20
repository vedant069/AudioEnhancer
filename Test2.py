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
        deepgram = DeepgramClient(api_key=deepgram_api_key)
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
    
    For each utterance, keep:
      - "start", "end", "transcript"
      - "words": a list of word objects each with only:
            "word", "start", and "end"
    """
    # Convert Deepgram response to dictionary if needed.
    if hasattr(transcription, 'to_dict'):
        transcription = transcription.to_dict()

    reduced = {"utterances": []}
    
    channels = transcription.get("results", {}).get("channels", [])
    if not channels:
        return reduced
        
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return reduced
        
    words = alternatives[0].get("words", [])
    current_utterance = None
    
    for word in words:
        minimal_word = {
            "word": word.get("word"),
            "start": word.get("start"),
            "end": word.get("end")
        }
        if current_utterance is None:
            current_utterance = {
                "start": word.get("start"),
                "end": word.get("end"),
                "transcript": word.get("word"),
                "words": [minimal_word]
            }
        else:
            # Start a new utterance if there's a long pause (>1 sec) or punctuation.
            if word.get("start") - current_utterance["end"] > 1.0 or \
               word.get("punctuated_word", "").endswith((".", "!", "?")):
                reduced["utterances"].append(current_utterance)
                current_utterance = {
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "transcript": word.get("word"),
                    "words": [minimal_word]
                }
            else:
                current_utterance["end"] = word.get("end")
                current_utterance["transcript"] += " " + word.get("word")
                current_utterance["words"].append(minimal_word)
    
    if current_utterance is not None:
        reduced["utterances"].append(current_utterance)
    
    return reduced

############################################################
# 4. FILTER TRANSCRIPTION WITH GEMINI-2.0-FLASH (Google GenAI)
############################################################
def filter_transcription_with_gemini(reduced_json, gemini_api_key):
    """
    Pass the reduced transcription JSON to Gemini-2.0-Flash with full context.
    
    The JSON contains for each utterance:
      - "start": utterance start time (in seconds)
      - "end": utterance end time (in seconds)
      - "transcript": the full text of the utterance
      - "words": a list of word objects (each with "word", "start", and "end")
    
    Using the detailed word-level timing, Gemini should analyze the transcription
    and remove duplicate or erroneous content (for example, duplicate self-introductions
    or repeated phrases) so that only the perfect output remains.
    
    Return a JSON object with two keys:
      1. "filtered_transcription": the updated transcription JSON with only the correct,
         de-duplicated speech.
      2. "removal_segments": a list of [start, end] time segments (in seconds) corresponding
         to parts of the audio that should be removed.
    
    Do not include any additional commentary. Use the input JSON exactly as provided.
    """
    prompt = f"""You are an expert audio transcription editor for voice-over recordings. You are given a JSON object representing a transcription. The JSON object has a key "utterances", which is a list of utterance objects. Each utterance object contains:
- "start": the start time of the utterance (in seconds),
- "end": the end time of the utterance (in seconds),
- "transcript": a string representing the full text of the utterance,
- "words": a list of word objects, where each word object has:
    - "word": the transcribed word,
    - "start": the start time of the word (in seconds),
    - "end": the end time of the word (in seconds).

Using the detailed word-level timing, analyze the transcription to identify and remove any duplicate or erroneous content (for example, repeated phrases or duplicate self-introductions), ensuring that only the perfect version remains. Preserve the original timestamps for the kept content.Remove the smallest of smallest duplication.when there are multiple versions of the same content, keep only the last version of the content. For example, if the transcription contains the following:
for example -and, and 
-like, like
-are, are
-for this , for this ..
etc
Be strict on removal of duplications and keep only the perfect version of the content.
Return a JSON object with two keys:
1. "filtered_transcription": the updated transcription JSON (with key "utterances") containing only the correct, de-duplicated speech.
2. "removal_segments": a list of [start, end] time segments (in seconds) corresponding to parts of the audio that should be removed.
Remember to keep the perfect version of the content and remove the smallest of smallest duplication.Take the last only because thats mostly the perfect one.

Do not include any extra commentary. Use the input JSON exactly as provided.

Input JSON:
{json.dumps(reduced_json, indent=2)}
"""
    try:
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
        )
        cleaned_text = re.sub(r'^```json\n|```$', '', response.text.strip())
        result = json.loads(cleaned_text)
        print("Gemini response:")
        print(json.dumps(result, indent=2))
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
