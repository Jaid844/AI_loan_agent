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
        "thread_id": "4",

    }
}

qn = [
      "Yeah I would like a loan adjustment"]
for question in qn:
    for output in app.stream({"messages": question, "session_id": "6=","name":"James"}, config, ):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
