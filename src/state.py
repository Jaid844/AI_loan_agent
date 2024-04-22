from typing import Annotated, Dict, TypedDict

from typing import List

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    transcription:str
    generation: str
    name: str
    session_id:str
    Profile:List[str]
