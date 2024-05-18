from langgraph.graph import END, StateGraph
from state import GraphState
from nodes import Nodes
fff

class WorkFlow():
    def __init__(self):
        nodes = Nodes()
        workflow = StateGraph(GraphState)
        # ADDING NODES
        workflow.add_node("profile_summarizer", nodes.customer_profile_summarizer)
        workflow.add_node("Good_customer_voice_1", nodes.customer_voice_1)
        workflow.add_node("Bad_customer_voice_2", nodes.customer_voice_1)
        workflow.add_node("Good_Profile_Chain", nodes.Good_Profile_Chain)
        workflow.add_node("Bad_Profile_Chain", nodes.Bad_Profile_Chain)
        workflow.add_node("Loan_Adjustment_Agent", nodes.loan_adjustment_agent)
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
        workflow.add_edge("Good_customer_voice_1", "Good_Profile_Chain")
        # LOAN MODIFICATION
        workflow.add_conditional_edges(
            "Good_customer_voice_1",
            nodes.grade_loan_adjustment,
            {
                "Loan_Adjustment_Agent": "Loan_Adjustment_Agent",
                "ai_voice": "Good_Profile_Chain"
            },
        )
        workflow.add_edge("Good_Profile_Chain", "Good_customer_voice_1")
        workflow.add_edge("Loan_Adjustment_Agent", "Good_Profile_Chain")
        # ENDING CONVERSATION
        workflow.add_conditional_edges(
            "Good_Profile_Chain",
            nodes.grade_conversation,
            {
                "END": END,
                "customer_voice": "Good_customer_voice_1"
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
