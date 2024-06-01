import uuid

from openai import OpenAI
from langchain_core.messages import HumanMessage
from graph import WorkFlow
from pprint import pprint

from src.tools import _print_event
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
# input_message = HumanMessage(content="yes i am willing to pay some portion of the loan amount")
qn = [HumanMessage(content="hellow"),
    HumanMessage(content="Yes i have some portion of the loan amount this month"),
    HumanMessage(content="yes i will pay some portion of the loan amount"),
    HumanMessage(content="my first name is James"),
      ]
for i in qn:
    for event in app.stream(
            {"human_messages": i, "name": "James"}, config,
    ):
        for key, value in event.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
    pprint(value["messages"])
