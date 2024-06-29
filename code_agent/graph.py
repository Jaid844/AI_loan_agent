from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from state import State
from nodes import *
from tools import *
from langgraph.checkpoint.memory import MemorySaver

cal_tools = [monthly_payment]

memory = MemorySaver()


class WorkFlow:
    def __init__(self):
        nodes = Nodes()
        workflow = StateGraph(State)
        # ADDING NODES
        workflow.add_node("profile_summarizer", nodes.customer_profile_summarizer)
        workflow.set_entry_point("profile_summarizer")
        workflow.add_node("primary_assistant", nodes.primary_assistant)
        workflow.add_node("bad_profile", nodes.Bad_Profile_Chain)
        workflow.add_node("enter_loan_tool", nodes.create_entry_node("Loan_calculator_assistant", "update_loan"))
        workflow.add_node("update_loan", Assistant(nodes.tool_runnable()))
        workflow.add_edge("enter_loan_tool", "update_loan")
        workflow.add_node("tool_use", create_tool_node_with_fallback(cal_tools))
        workflow.add_edge("tool_use", "update_loan")
        workflow.set_entry_point("profile_summarizer")
        workflow.add_conditional_edges(
            "profile_summarizer",
            nodes.grade_profile,
            {
                "primary_assistant": "primary_assistant",
                "bad_profile": "bad_profile",
            },
        )
        workflow.add_edge("bad_profile",END)
        workflow.add_conditional_edges(
            "update_loan",
            route_to_tool,
            {
                "leave_skill": "leave_skill",
                END: END,
                "tool_use": "tool_use"
            }
        )
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
        #workflow.add_conditional_edges("user_profile", route_to_workflow)

        self.app = workflow.compile(checkpointer=memory)
