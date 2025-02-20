import os
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from google import genai
import re

#############################################
# 1. AUDIO RECORDING & SAVING
#############################################
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

#############################################
# 2. AUDIO TRANSCRIPTION (Deepgram)
#############################################
def transcribe_audio(filepath, deepgram_api_key):
    """
    Transcribe an audio file using Deepgram Nova.
    """
    try:
        deepgram = DeepgramClient(api_key='a8b75fa07ad77e26a7866d995ed329553927767b')
        with open(filepath, 'rb') as file:
            buffer_data = file.read()
        payload: FileSource = {"buffer": buffer_data}
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
        
        # Debug: print the Deepgram response structure.
        print("Deepgram response structure:")
        print(json.dumps(response.to_dict(), indent=2))
        
        return response
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

#############################################
# 3. REDUCE TRANSCRIPTION JSON TO ESSENTIAL FIELDS
#############################################
def reduce_transcription_json(transcription):
    """
    Reduce the transcription JSON by keeping only the necessary fields.
    Group words into utterances with:
      - "start", "end", "transcript"
      - "words": list of word objects (each with "word", "start", and "end")
    """
    # Convert to dictionary if needed.
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
        if current_utterance is None:
            current_utterance = {
                "start": word.get("start"),
                "end": word.get("end"),
                "transcript": word.get("word"),
                "words": [{
                    "word": word.get("word"),
                    "start": word.get("start"),
                    "end": word.get("end")
                }]
            }
        else:
            # Start a new utterance if a long pause (>1 sec) or punctuation indicates a break.
            if word.get("start") - current_utterance["end"] > 1.0 or \
               word.get("punctuated_word", "").endswith((".", "!", "?")):
                reduced["utterances"].append(current_utterance)
                current_utterance = {
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "transcript": word.get("word"),
                    "words": [{
                        "word": word.get("word"),
                        "start": word.get("start"),
                        "end": word.get("end")
                    }]
                }
            else:
                current_utterance["end"] = word.get("end")
                current_utterance["transcript"] += " " + word.get("word")
                current_utterance["words"].append({
                    "word": word.get("word"),
                    "start": word.get("start"),
                    "end": word.get("end")
                })
    if current_utterance is not None:
        reduced["utterances"].append(current_utterance)
    return reduced

#############################################
# 4. FILTER TRANSCRIPTION WITH GEMINI-2.0-FLASH (Google GenAI)
#############################################
def filter_transcription_with_gemini(reduced_json, gemini_api_key):
    """
    Pass the reduced transcription JSON to Gemini-2.0-Flash to remove duplicate
    or erroneous content. The prompt instructs Gemini to:
      - Keep only the final (correct) self-introduction and remove duplicate utterances.
      - Return a JSON object with:
            "filtered_transcription": the updated transcription (with key "utterances")
            "removal_segments": list of [start, end] segments (in seconds) to remove.
    """
    prompt = f"""You are an audio transcription editor for voice-over recordings. You are given a JSON object representing a transcription. The JSON object has a key "utterances", which is a list of utterance objects. Each utterance object contains:
- "start": the start time (in seconds),
- "end": the end time (in seconds),
- "transcript": the full text of the utterance,
- "words": a list of word objects (each having "word", "start", and "end").

The speaker has accidentally recorded duplicate audio. For example, if there are multiple utterances like "Hi. My name is Vedant.", only the final self-introduction is correct. Remove any previous self-introductions or duplicated utterances and keep only the correct, unique content, preserving the timestamps. Also, remove extraneous fields so that only the mentioned fields remain. Ensure there are no duplications.

Return a JSON object with two keys:
1. "filtered_transcription": the updated transcription JSON (with key "utterances") containing only the necessary utterances.
2. "removal_segments": a list of [start, end] time segments (in seconds) corresponding to parts of the audio that should be removed.

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
        cleaned_text = re.sub(r'^```json\n|```$', '', response.text.strip())
        result = json.loads(cleaned_text)
        print("Gemini response:")
        # print(json.dumps(result, indent=2))
        print(response)
        return result
    except Exception as e:
        print(f"Error processing transcription with Gemini: {e}")
        return None

#############################################
# 5. EXTRACTION OF FILLER & PAUSE REMOVAL SEGMENTS
#############################################
def get_filler_removal_segments(transcription, pause_threshold=0.5, filler_words={"um", "uh", "hmm", "erm", "ah"}):
    """
    From the original Deepgram transcript, extract segments corresponding to filler words
    and long pauses (gaps between words greater than pause_threshold).
    
    Returns a list of (start, end) segments (in seconds) to be removed.
    """
    if hasattr(transcription, 'to_dict'):
        transcription = transcription.to_dict()
    try:
        words = transcription["results"]["channels"][0]["alternatives"][0]["words"]
    except Exception as e:
        print(f"Error extracting words: {e}")
        return []
    
    removal_segments = []
    for word in words:
        if word["word"].lower() in filler_words:
            removal_segments.append((word["start"], word["end"]))
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        gap = next_word["start"] - current_word["end"]
        if gap > pause_threshold:
            removal_segments.append((current_word["end"], next_word["start"]))
    return removal_segments

#############################################
# 6. EXTRACTION OF PROFANITY SEGMENTS
#############################################
def get_profanity_segments(transcription, profanity_set={
    # Common profanity
    "fuck", "fucker", "fucking", "shit", "damn", "bitch", "asshole", "cunt",
    # Racial and ethnic slurs
    "nigga", "nigger", "niggas", "chink", "spic", "kike", "wetback", "gook",
    # Additional profanity
    "ass", "bastard", "dick", "pussy", "whore", "slut", "cock", "piss",
    # Sexual references
    "blowjob", "handjob", "rimjob", "fellatio", "cunnilingus",
    # Compound words and phrases
    "motherfucker", "bullshit", "dumbass", "jackass", "dickhead", "motherfucking",
    "son of a bitch", "piece of shit", "fuck you", "fuck off",
    # Religious profanity
    "goddamn", "jesus christ", "christ sake", "holy shit", "god damn",
    # Milder profanity and insults
    "hell", "dammit", "damn it", "shut up", "idiot", "moron", "retard", "retarded",
    # Common variations and misspellings
    "fck", "fuk", "fuking", "shit", "bitch", "asshole", "fuck",
    # Compound variations
    "af", "wtf", "stfu", "gtfo", "lmfao", "pos",
    # Body parts used as insults
    "tits", "boobs", "penis", "vagina", "balls"
}):
    """
    From the Deepgram transcript, extract segments corresponding to profanity words.
    Returns a list of (start, end) segments (in seconds) that should be beeped.
    """
    if hasattr(transcription, 'to_dict'):
        transcription = transcription.to_dict()
    try:
        words = transcription["results"]["channels"][0]["alternatives"][0]["words"]
    except Exception as e:
        print(f"Error extracting words for profanity: {e}")
        return []
    
    profanity_segments = []
    for word in words:
        if word["word"].lower() in profanity_set:
            profanity_segments.append((word["start"], word["end"]))
    return profanity_segments

#############################################
# 7. MERGE SEGMENTS (for a single type)
#############################################
def merge_segments(segments):
    """
    Merge overlapping or contiguous segments.
    segments: list of (start, end)
    Returns a merged list.
    """
    if not segments:
        return []
    segments.sort(key=lambda x: x[0])
    merged = [segments[0]]
    for seg in segments[1:]:
        last_seg = merged[-1]
        if seg[0] <= last_seg[1]:
            merged[-1] = (last_seg[0], max(last_seg[1], seg[1]))
        else:
            merged.append(seg)
    return merged

#############################################
# 8. ENHANCE AUDIO WITH REPLACEMENT SEGMENTS
#############################################
def generate_beep(duration, sample_rate, frequency=1000, amplitude=0.3):  # Reduced amplitude from 0.7 to 0.3
    """
    Generate a beep tone (sine wave) for a given duration.
    Returns a numpy array of int16 samples.
    
    Args:
        duration (float): Duration of the beep in seconds
        sample_rate (int): Sample rate in Hz
        frequency (int): Frequency of the beep tone in Hz (default: 1000)
        amplitude (float): Amplitude of the sine wave (default: 0.3, reduced from 0.7)
    """
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    # Apply a slight envelope to avoid clicking
    envelope = np.minimum(t * 100, np.minimum(1, (duration - t) * 100))
    tone = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    # Scale to int16 range
    beep = np.int16(tone / np.max(np.abs(tone)) * 32767)
    return beep

def enhance_audio_with_replacement(audio_filepath, replacement_segments, output_filename="enhanced_audio.wav"):
    """
    Process the audio file by removing or replacing segments.
    replacement_segments: list of tuples (start, end, seg_type) where seg_type is "remove" or "beep".
      - For "remove" segments, the audio is omitted.
      - For "beep" segments, a beep tone of the same duration is inserted.
    """
    try:
        sample_rate, data = wav.read(audio_filepath)
        if data.ndim > 1:
            data = data[:, 0]
        # Convert replacement segments from seconds to sample indices.
        segs = []
        for start, end, seg_type in replacement_segments:
            segs.append((int(start * sample_rate), int(end * sample_rate), seg_type))
        # Sort segments by start index.
        segs.sort(key=lambda x: x[0])
        
        output_audio = []
        current_idx = 0
        for start_idx, end_idx, seg_type in segs:
            # Append audio from current index to start of segment.
            if start_idx > current_idx:
                output_audio.append(data[current_idx:start_idx])
            seg_duration = (end_idx - start_idx) / sample_rate
            if seg_type == "beep":
                # Generate beep tone for the duration.
                beep_tone = generate_beep(seg_duration, sample_rate)
                output_audio.append(beep_tone)
            # For "remove", we simply do not add the segment.
            current_idx = end_idx
        if current_idx < len(data):
            output_audio.append(data[current_idx:])
        if output_audio:
            enhanced_audio = np.concatenate(output_audio)
        else:
            enhanced_audio = data
        wav.write(output_filename, sample_rate, enhanced_audio)
        print(f"Enhanced audio saved to {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error during audio enhancement with replacement: {e}")
        return None

#############################################
# 9. MAIN PIPELINE
#############################################
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
        
        # 2. Transcribe audio using Deepgram.
        transcription = transcribe_audio(audio_file, deepgram_api_key)
        if transcription is None:
            print("Transcription failed. Exiting.")
            return
        
        # 3. Reduce the transcription.
        reduced_json = reduce_transcription_json(transcription)
        
        # 4. Use Gemini to filter duplicates/erroneous content.
        gemini_result = filter_transcription_with_gemini(reduced_json, gemini_api_key)
        if gemini_result is None:
            print("Gemini filtering failed. Exiting.")
            return
        gemini_removal_segments = gemini_result.get("removal_segments", [])
        print("Gemini removal segments:")
        print(gemini_removal_segments)
        
        # 5. Extract filler word and pause removal segments.
        filler_removal_segments = get_filler_removal_segments(transcription, pause_threshold=0.5)
        print("Filler removal segments:")
        print(filler_removal_segments)
        
        # 6. Extract profanity segments.
        profanity_segments = get_profanity_segments(transcription)
        print("Profanity segments:")
        print(profanity_segments)
        
        # 7. Mark replacement segments with type.
        # "remove" segments come from Gemini and filler removal.
        replacement_segments = []
        for seg in gemini_removal_segments:
            replacement_segments.append((seg[0], seg[1], "remove"))
        for seg in filler_removal_segments:
            replacement_segments.append((seg[0], seg[1], "remove"))
        # "beep" segments for profanity.
        for seg in profanity_segments:
            replacement_segments.append((seg[0], seg[1], "beep"))
        
        # Merge segments separately for each type.
        remove_segs = merge_segments([(s, e) for s, e, t in replacement_segments if t=="remove"])
        beep_segs = merge_segments([(s, e) for s, e, t in replacement_segments if t=="beep"])
        
        # Recombine with type information.
        final_replacement_segments = []
        for s, e in remove_segs:
            final_replacement_segments.append((s, e, "remove"))
        for s, e in beep_segs:
            final_replacement_segments.append((s, e, "beep"))
        # Sort by start time.
        final_replacement_segments.sort(key=lambda x: x[0])
        print("Final replacement segments (start, end, type):")
        print(final_replacement_segments)
        
        # 8. Enhance audio by applying the replacement segments.
        enhanced_file = enhance_audio_with_replacement(audio_file, final_replacement_segments, output_filename="enhanced_audio.wav")
        
        # Optionally, remove the original audio file:
        # os.remove(audio_file)
        
    except Exception as e:
        print(f"Exception in main: {e}")

if __name__ == "__main__":
    main()
