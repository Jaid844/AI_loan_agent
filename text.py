import sys
import requests
import pyaudio
import soundfile as sf
import io
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


starttime=time.time()
client = OpenAI()
audio_file= open("myrecording.wav", "rb")
transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
transcription=transcription.text
print(f"Time to play: {time.time() - starttime} seconds")
print(transcription)
