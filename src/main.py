import uuid

from langgraph.checkpoint.sqlite import SqliteSaver

from graph import WorkFlow
from pprint import pprint

from src.tools import _print_event
from tools import *

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
qn = ["Hi what is this call about",
      "Yeah I would like a loan adjusment"]
for question in qn:
    events = app.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
