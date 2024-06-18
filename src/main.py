import os.path
import uuid

from groq import Groq
from openai import OpenAI

from graph import WorkFlow
from pprint import pprint
from audio_recorder_streamlit import audio_recorder
import streamlit as st
import soundfile as sf
from src.tools import _print_event
from tools import *

load_dotenv()
client = OpenAI()
app = WorkFlow().app
thread_id = str(uuid.uuid4())
_printed = set()

config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": "29-",

    }
}
_printed = set()

qn = [
"yeah a loan adjustment would be great",
    "No the loan adjustment seem to step for me",
]
for question in qn:
    events = app.stream(
        {"messages": ("user", question), "name": "James"}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)