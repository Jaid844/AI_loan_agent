import uuid

from openai import OpenAI

from graph import WorkFlow
from pprint import pprint
from audio_recorder_streamlit import audio_recorder
import streamlit as st
import soundfile as sf
from tools import *

client = OpenAI()
app = WorkFlow().app
thread_id = str(uuid.uuid4())
_printed = set()

config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
        "name": "James",
    }
}

st.title("Call recorder")
recorded_audio = audio_recorder()
if recorded_audio:
    audiofile = "audio.wav"
    with open(audiofile, 'wb') as f:
        f.write(recorded_audio)
    audio_file1 = open(audiofile, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file1
    )
    for event in app.stream(
            {"messages": ("user", transcription.text)}, config, stream_mode="values"
    ):
        for key, value in event.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
    print("finished")
