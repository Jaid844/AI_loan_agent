from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from state import State
from nodes import *
from tools import *
from langgraph.checkpoint.memory import MemorySaver
tools = [monthly_payment]

memory = MemorySaver()
class WorkFlow():
    def __init__(self):
        nodes = Nodes()
        workflow = StateGraph(State)
        # ADDING NODES
        # workflow.add_node("user_profile", nodes.customer_profile_summarizer)
        workflow.set_entry_point("primary_assistant")
        workflow.add_node("primary_assistant", Assistant(nodes.primary_assistant()))
        workflow.add_node("enter_loan_tool", nodes.create_entry_node("Loan_calculator_assistant", "update_loan"))
        workflow.add_node("update_loan", Assistant(nodes.tool_runnable()))
        workflow.add_edge("enter_loan_tool", "update_loan")
        workflow.add_node("tool_use", nodes.create_tool_node_with_fallback(tools))
        workflow.add_edge("tool_use", "update_loan")
        workflow.add_conditional_edges("update_loan", route_to_tool)
        workflow.add_node("leave_skill", pop_dialog_state)
        workflow.add_edge("leave_skill", "primary_assistant")
        workflow.add_conditional_edges(
            "primary_assistant",
            route_primary_assistant,
            {
                "enter_loan_tool": "enter_loan_tool",
                END: END
            }
        )
        # workflow.add_conditional_edges("user_profile", route_to_workflow)

        self.app = workflow.compile(checkpointer=memory)
