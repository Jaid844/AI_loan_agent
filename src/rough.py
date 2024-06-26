from itertools import cycle
from typing import Literal, Union

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from typing_extensions import Annotated

from pydantic import BaseModel, Discriminator, Field, Tag

from importlib import util
infinite_cycle = cycle([AIMessage(content="hello"), AIMessage(content="goodbye")])
model = GenericFakeChatModel(messages=infinite_cycle)
response = model.invoke("kitty")
print(response)