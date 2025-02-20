# main.py (python example)

import os
import logging
from deepgram.utils import verboselogs

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)

AUDIO_URL = {
    "url": "https://dpgr.am/bueller.wav"
}

def main():
    try:
        # STEP 1 Create a Deepgram client using the DEEPGRAM_API_KEY from your environment variables
        deepgram: DeepgramClient = DeepgramClient(api_key='a8b75fa07ad77e26a7866d995ed329553927767b')

        # STEP 2 Call the transcribe_url method on the rest class
        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            filler_words = True,
            profanity_filter = True,
            utterances = True,
            punctuate=True,
            utt_split=1
        )
        response = deepgram.listen.rest.v("1").transcribe_url(AUDIO_URL, options)
        print(f"response: {response}\n\n")

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()