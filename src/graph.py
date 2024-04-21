from langgraph.graph import END, StateGraph
from state import GraphState
from nodes import Nodes

class WorkFlow():
    def __init__(self):
        nodes = Nodes()
        workflow = StateGraph(GraphState)
        ## ADDING NODES
        workflow.add_node("customer_voice",nodes.customer_voice)
        workflow.add_node("ai_voice", nodes.ai_voice)

        ## STITCHING NODES
        workflow.set_entry_point("customer_voice")
        workflow.add_conditional_edges(
            "ai_voice",
            nodes.grade_conversation,
            {
                "customer_voice": "customer_voice",
                "END": END,
            },
        )
        workflow.add_edge("customer_voice", "ai_voice")
        self.app=workflow.compile()