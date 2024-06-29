import os.path
import uuid

from groq import Groq
from openai import OpenAI

from graph import WorkFlow
from pprint import pprint
from audio_recorder_streamlit import audio_recorder
import streamlit as st
import soundfile as sf
from code_agent.tools import _print_event
from tools import *

load_dotenv()
client = OpenAI()
app = WorkFlow().app
thread_id = str(uuid.uuid4())
_printed = set()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Loan Agent adjustment debug"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": "2ep-",

    }
}
_printed = set()

qn = [
"hellow",
"yeah a loan adjustment would be great",
"seems good to me",
"Still the adjustment seem to be more steep"
]
for question in qn:
    events = app.stream(
        {"messages": ("user", question), "name": "James"}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)