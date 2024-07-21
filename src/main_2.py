from graph import WorkFlow
import streamlit as st
from pydub import AudioSegment
import io
from audio_recorder_streamlit import audio_recorder
from tools import *

load_dotenv()
client = OpenAI()

app = WorkFlow().app
# Title of the App
st.title("Call recorder")

# Thread ID
text_input_container_2 = st.empty()
thread = text_input_container_2.text_input("Thread_id")

# Config Thread id of the Run

config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread,
    }
}

# Name of the customer
text_input_container = st.empty()
name = text_input_container.text_input("Enter the name of the Person")

# Deleting the old audio
asyncio.run(remove_audio("audio.wav"))

# Recording the audio
recorded_audio = audio_recorder()

# Displaying the Name of the customer in the UI
if name != "":
    text_input_container.empty()
    st.info(name)

# If recorded audio and name given in the session then we run the App
if recorded_audio and name:
    # Record the Audio
    audio_file1 = asyncio.run(transcribe_audio("audio.wav", recorded_audio))

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file1
    )
    app.invoke(
        {"messages": ("user", transcription.text), "name": name}, config, stream_mode="values"
    )
