import os
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

def record_audio(duration=5, sample_rate=44100):
    """
    Record audio from the microphone.
    
    Args:
        duration (int): Recording duration in seconds.
        sample_rate (int): Sample rate for audio recording.
    
    Returns:
        numpy.ndarray: Recorded audio data.
        int: Sample rate.
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
    
    Args:
        audio_data (numpy.ndarray): Audio data to save.
        sample_rate (int): Sample rate of the audio.
        filename (str): Output filename.
    
    Returns:
        str: The filename that was saved.
    """
    wav.write(filename, sample_rate, audio_data)
    return filename

def transcribe_audio(filepath, api_key):
    """
    Transcribe audio file using Deepgram.
    
    Args:
        filepath (str): Path to the audio file.
        api_key (str): Deepgram API key.
    
    Returns:
        dict: Transcription response.
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
        print(response)
        return response
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def enhance_audio(audio_filepath, transcript, output_filename="enhanced_audio.wav", pause_threshold=0.5):
    """
    Enhance the audio by removing filler words and long pauses.
    
    This function:
      - Reads the WAV file.
      - Parses the transcript to extract the timestamps for filler words
        and long pauses (gaps between words longer than pause_threshold).
      - Removes these segments and stitches together the remaining audio.
      - Saves the enhanced audio to a new file.
    
    Args:
        audio_filepath (str): Path to the original audio file.
        transcript (dict): Deepgram transcription response.
        output_filename (str): Filename for the enhanced audio.
        pause_threshold (float): Minimum gap (in seconds) to consider as a pause for removal.
    
    Returns:
        str: The filename of the enhanced audio.
    """
    try:
        # Read the audio data from the file
        sample_rate, data = wav.read(audio_filepath)
        # In case the data is stereo or has extra dimensions, ensure it is one-dimensional.
        if data.ndim > 1:
            data = data[:, 0]
        
        # Extract the word-level details from the transcript
        words = transcript["results"]["channels"][0]["alternatives"][0]["words"]
        
        removal_segments = []
        # Define the set of filler words you want to remove
        filler_words = {"um", "uh", "hmm", "erm", "ah"}  # adjust as needed
        
        # Mark segments for filler words
        for word in words:
            if word["word"].lower() in filler_words:
                removal_segments.append( (word["start"], word["end"]) )
        
        # Mark long pauses (gaps between words)
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            gap = next_word["start"] - current_word["end"]
            if gap > pause_threshold:
                removal_segments.append( (current_word["end"], next_word["start"]) )
        
        # If no removal segments are found, keep the audio as is
        if not removal_segments:
            print("No filler words or long pauses detected. No enhancement performed.")
            return audio_filepath
        
        # Merge overlapping or contiguous segments
        removal_segments.sort(key=lambda seg: seg[0])
        merged_segments = []
        for seg in removal_segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_seg = merged_segments[-1]
                # If segments overlap or are contiguous, merge them
                if seg[0] <= last_seg[1]:
                    new_seg = (last_seg[0], max(last_seg[1], seg[1]))
                    merged_segments[-1] = new_seg
                else:
                    merged_segments.append(seg)
        
        # Convert time segments (in seconds) to sample indices
        removal_indices = [(int(start * sample_rate), int(end * sample_rate)) for start, end in merged_segments]
        
        # Build a list of segments to keep
        keep_segments = []
        current_idx = 0
        for start_idx, end_idx in removal_indices:
            if start_idx > current_idx:
                keep_segments.append(data[current_idx:start_idx])
            current_idx = end_idx
        # Append any remaining audio after the last removal segment
        if current_idx < len(data):
            keep_segments.append(data[current_idx:])
        
        if keep_segments:
            enhanced_audio = np.concatenate(keep_segments)
        else:
            enhanced_audio = data  # Fallback if nothing remains
        
        # Save the enhanced audio to a new file
        wav.write(output_filename, sample_rate, enhanced_audio)
        print(f"Enhanced audio saved to {output_filename}")
        return output_filename

    except Exception as e:
        print(f"Error during audio enhancement: {e}")
        return None

def main():
    try:
        # Replace with your Deepgram API key
        api_key = "a8b75fa07ad77e26a7866d995ed329553927767b"
        
        duration = float(input("Enter recording duration in seconds: "))
        
        # 1. Record and save audio
        audio_data, sample_rate = record_audio(duration=duration)
        audio_file = save_audio(audio_data, sample_rate)
        print(f"Audio saved to: {audio_file}")
        
        # 2. Transcribe the audio
        transcript = transcribe_audio(audio_file, api_key)
        if transcript is None:
            print("Transcription failed. Exiting.")
            return
        
        # For debugging purposes, you can print the transcript JSON:
        # print(json.dumps(transcript, indent=2))
        
        # 3. Enhance the audio by removing filler words and long pauses
        enhanced_file = enhance_audio(audio_file, transcript, output_filename="enhanced_audio.wav", pause_threshold=0.5)
        
        # Optionally, you can remove the original file if desired:
        # os.remove(audio_file)
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()
