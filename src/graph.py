from langgraph.graph import END, StateGraph
from state import State
from nodes import Nodes, create_entry_node, Assistant, create_tool_node_with_fallback, monthly_payment, \
    pop_dialog_state, \
    route_in, loan_tool_chain, route_primary_assistant


class WorkFlow():
    def __init__(self):
        nodes = Nodes()
        workflow = StateGraph(State)
        # ADDING NODES
        workflow.add_node("enter_loan_tool", create_entry_node("Loan_assitant", "Loan_Calculator_chain"))
        workflow.add_node("Loan_Calculator_chain", Assistant(loan_tool_chain))
        workflow.add_node("tool_use_loan", create_tool_node_with_fallback([monthly_payment]))
        workflow.add_node("leave_skill", pop_dialog_state)
        workflow.add_edge("enter_loan_tool", "Loan_Calculator_chain")
        workflow.add_edge("tool_use_loan", "Loan_Calculator_chain")
        workflow.add_edge("leave_skill", "Primary_Good_Profile_Chain")

        workflow.add_node("profile_summarizer", nodes.customer_profile_summarizer)
        workflow.add_node("Good_customer_voice_1", nodes.customer_voice_1)
        workflow.add_node("Bad_customer_voice_2", nodes.customer_voice_1)
        workflow.add_node("Primary_Good_Profile_Chain", nodes.Good_Profile_Chain)
        workflow.add_node("Bad_Profile_Chain", nodes.Bad_Profile_Chain)

        # for good chain

        workflow.add_edge("Loan_Calculator_chain", "Primary_Good_Profile_Chain")
        # STITCHING NODES
        workflow.set_entry_point("profile_summarizer")
        workflow.add_conditional_edges(
            "profile_summarizer",
            nodes.grade_profile,
            {
                "Good_Profile_Voice": "Good_customer_voice_1",
                "Bad_Profile_Voice": "Bad_customer_voice_2",
            },
        )
        # GOOD PROFILE VOICE 'GREETING'
        workflow.add_edge("Good_customer_voice_1", "Primary_Good_Profile_Chain")
        workflow.add_conditional_edges(
            "Primary_Good_Profile_Chain",
            route_primary_assistant,
            {
                "enter_loan_tool": "enter_loan_tool",
                "Good_customer_voice_1": "Good_customer_voice_1",
                END: END
            }
        )

        # ENDING CONVERSATION
        workflow.add_conditional_edges(
            "Primary_Good_Profile_Chain",
            nodes.grade_conversation,
            {
                "END": END,
                "customer_voice": "Good_customer_voice_1"
            }

        )
        workflow.add_conditional_edges(
            "Loan_Calculator_chain",
            route_in,
            {
                "leave_skill": "leave_skill",
                "tool_use_loan": "tool_use_loan"

            }
        )
        workflow.add_edge("Bad_customer_voice_2", "Bad_Profile_Chain")
        workflow.add_conditional_edges(
            "Bad_Profile_Chain",
            nodes.grade_conversation,
            {
                "customer_voice": "Bad_customer_voice_2",
                "END": END
            },
        )

        self.app = workflow.compile()
