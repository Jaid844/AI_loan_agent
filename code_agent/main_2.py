import os.path


from groq import Groq
from openai import OpenAI

from graph import WorkFlow

from audio_recorder_streamlit import audio_recorder
import streamlit as st

from tools import *

load_dotenv()
client = OpenAI()

app = WorkFlow().app
st.title("Call recorder")
text_input_container_2 = st.empty()
thread = text_input_container_2.text_input("Thread_id")
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread,
        "name": "James",
    }
}
text_input_container = st.empty()
name = text_input_container.text_input("Enter the name of the Person")
audiofile = "audio.wav"
if os.path.exists(audiofile):
    os.remove(audiofile)
recorded_audio = audio_recorder()

if name != "":
    text_input_container.empty()
    st.info(name)

if recorded_audio and name:
    audiofile = "audio.wav"
    with open(audiofile, 'wb') as f:
        f.write(recorded_audio)
    audio_file1 = open(audiofile, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file1
    )
    app.invoke(
        {"messages": ("user", transcription.text), "name": name}, config, stream_mode="values"
    )
