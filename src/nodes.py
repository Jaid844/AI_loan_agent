from typing import Callable, Union
from typing import Literal
from langgraph.graph import END
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from tools import *
from state import State
from langchain_core.pydantic_v1 import BaseModel, Field
from audio import audio_node
import os
load_dotenv()
embeddings = VoyageAIEmbeddings(
    model="voyage-2", batch_size=128, truncation=True
)
tools = [monthly_payment]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Loan Agent adjustment"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    reason: str

    class Config:
        schema_extra = {
            "example 1": {
                "reason": """The calculated loan amount for this month, after applying a 5% discount, is $16.67. 
Would you like to proceed with this amount, James?""",
            },
            "example 2": {
                "reason": "I have fully calculated the loan amount", }
        }


from typing_extensions import Annotated


class loan_amount_5(BaseModel):
    rate: Literal[5]
    name: str = Field(
        description="The name of the customer "
    )
    dialogue: str = Field(
        description="The conversion of the customer if he agrees to pay some portion of the loan amount"
    )


class loan_amount_10(BaseModel):
    rate: Literal[10]
    name: str = Field(
        description="The name of the customer "
    )


class To_Loan_tool_1(BaseModel):
    """
    This function will be able to calculate the loan amount for the customer ,Initially the rate will be 5 %
    but if the customer is felling a bit steep pay,A 10 % rate will be calculated


      """

    rate: Union[loan_amount_5, loan_amount_10] = Field(discriminator="rate")

    class Config:
        schema_extra = {
            "example 1": {
                "rate": 5,
                "name": "jake",
                "dialogue": "I would like a loan adjustment",
            },
            "example 2": {
                "rate": 10,
                "name": "shela",
                "dialogue": "The loan adjustment is too steep for me",
            }
        }


class Assistant:

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)  # the input are converted into dictionary key value pair

            if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}

            else:
                break
        return {"messages": result}


class Nodes():
    def __init__(self):
        self.audio = audio_node()

    def customer_profile_summarizer(self, state):
        name = state['name']
        documents = loan_embedding_model().get_relevant_documents(name)
        llm = ChatOpenAI(model='gpt-4o-mini')
        prompt = PromptTemplate(
            template=""" Summarize the profile of the customer below ,summarize the how he is with loan payment,financial circumstance
                  ,communication ,his credit worthiness  as detail as possilbe \n
                  Here is the context {context}
                  """,
            input_variables=["context"], )
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": documents})
        return {
            "profile": generation,
        }

    def primary_assistant(self, state):
        llm = ChatOpenAI(model='gpt-4o-mini')
        # llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        messages = state['messages']
        name = state['name']
        session_id = state['session_id']
        system = """"You are loan agent called as Sandy from ABC bank here to discuss the loan payment this customer has 
                  a good payment history
                  This is going to be a telephonic call so play along have a small conversation                 
                Here is the name of the customer {name}                 
                You are given set of example so you can reference from it
                "INSTRUCTIONS"
                -"GREET THEM WITH HELLOW AND ASK THEM THE REASON  WHY DID NOT THEY PAID THIS MONTH PAYMENT"
                -Keep the conversation quite and small
                -ASK THEM IF THEM WILLING TO PAY SOME PORTION OF THE LOAN AMOUNT 
                -WHILE CALCULATING THE LOAN AMOUNT TELL THEM TO HOLD FOR A MINUTE SO THAT YOU CAN CALCULATE THE NEW 
                DISCOUNTED LOAN AMOUNT
                -AT FIRST INITIALLY YOU WILL GIVE THEM 5 % DISCOUNT IN THEIR OUTSTANDING LOAN AMOUNT TELL THEM THIS THE 
                AMOUNT THEY WILL PAY THIS MONTH FROM NOW
                -IF THEY HESITATE FOR 5 % LOAN AMOUNT ,YOU PROVIDE THEM WITH 10 % DISCOUNT RATE BECAUSE THEY ARE GOOD 
                PAYING CUSTOMER BUT REMEMBER THIS IS THE LAST RATE THEY GET ,NOT BEYOND 10%
                """
        primary_assistant_prompt = ChatPromptTemplate.from_messages(
            [("system", system),
             ("placeholder", "{messages}")]
        )
        # llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        primary_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
            [To_Loan_tool_1])

        generation = primary_assistant_runnable.invoke({"messages": messages, "name": name},
                                                       config={"configurable": {"session_id": session_id}})
        if generation.tool_calls:
            pass
        else:
            self.audio.streamed_audio(generation.content)
            #tool_calls = [tc['name'] for m in state['messages'] for tc in getattr(m, 'tool_calls', [])]
            #print(tool_calls)

        return {
            "messages": generation
        }

    def Bad_Profile_Chain(self, state):
        name = state['name']
        messages = state['messages']
        llm = ChatOpenAI(model="gpt-4o-mini")
        system = """
        you are a loan agent named Sandy from ABC bank ,here to talk to customer who have bad credit 
        Find out the reason why didn't he paid this month due,If he is not willing cooperate,
        tell him the bank will take legal action against him.Remember that you dont need to do any adjustment
        here is the profile{name}
        INSTRUCTIONS
                -GREET THEM WITH HELLOW AND ASK THEM WHY DID THEY PAID THIS MONTH PAYMENT
                -ASK THEM WHY DIDNT THEY PAID THE LOAN AMOUNT THIS MONTH
                -IF THEY SLACK OR MISCOMMUNICATION  THREATEN WITH LAWSUIT


        """

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("placeholder", "{messages}"),
            ]
        )
        rag_chain = final_prompt | llm
        generation = rag_chain.invoke({"messages": messages, "name": name})
        self.audio.streamed_audio(generation)
        return {
            "messages": generation,
        }

    def grade_profile(self, state):
        profile = state['profile']

        class GradeConclusion(BaseModel):
            """Binary score for profile to see if the profile is good profile or the bad profile
            """

            binary_score: str = Field(
                description="Profile if they are good or bad based on credit history, 'Good' or 'Bad'")

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeConclusion)
        system = """You are a grader assessing the profiles of customer your job is to see if the credit score of the customer are good 
              or bad ,Grade 'Good' if the profile is Good ,or grade it Bad if the profile of the customer is 'Bad'
                          """
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Customer Profile: {profile}"),
            ]
        )
        customer_profile_grader = grade_prompt | structured_llm_grader

        score = customer_profile_grader.invoke({"profile": profile})
        if score.binary_score == "Good":
            return "primary_assistant"
        else:
            return "bad_profile"

    def tool_runnable(self):
        llm = ChatOpenAI(model='gpt-4o-mini')
        # llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        loan_hotel_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 " You are a specialized assistant for calculating loan amount of a customer "
                 " The primary assistant delegates work to you whenever the user needs help with calculating loan amount"
                 "  When searching, be persistent. Expand your query bounds if the first search returns no results. "
                 " Once you have calculated loan amount delegate back to  main assistant."
                 "  Remember that a loan amount  isn't completed until after the relevant tool has successfully been used."
                 "  then CompleteOrEscalate the dialog to the host assistant."
                 "  Do not waste the user's time. Do not make up invalid tools or functions."
                 " You dont need first name of the customer that's it"
                 "  then end the conversation by say such word as bye "
                 " You just need first name to calculate the loan amount "
                 " Remember to tell the user they will get 5% discount in their loan amount ,if they agree then that will"
                 "be their loan amount ,if they disagree offer them 10 % discount in their loan amount only if they "
                 "disagree in 5 % loan amount calculation"
                 "Only 5% and 10% loan adjustment is possible beyond that not possible"
                 " Name of the customer is {name}"
                 " The loan tool will tell how much amount will the customer will pay this month"
                 " If you have calculated the loan amount then use  CompleteOrEscalate function call /tool"
                 " \n\nSome examples for which you should CompleteOrEscalate:\n"""
                 " - The total loan amount will be $316.66. for James "
                 "- I HAVE FULLY CALCULATED THE LOAN AMOUNT"

                 ),
                ("placeholder", "{messages}")
            ]
        )
        tool_1 = [monthly_payment]
        loan_tool_runnable = loan_hotel_prompt | llm.bind_tools(tool_1 + [CompleteOrEscalate])
        return loan_tool_runnable

    def create_entry_node(self, assistant_name: str, new_dialog_state: str) -> Callable:
        def entry_node(state: State) -> dict:
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            return {
                "messages": [
                    ToolMessage(
                        content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                                f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                                " remember to  invoked the appropriate tool for calculating loan adjustment"
                                " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                                " Do not mention who you are - just act as the proxy for the assistant.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": new_dialog_state,
            }

        return entry_node


def route_to_tool(
        state: State,
) -> Literal[
    "tool_use",
    "leave_skill",
]:
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "tool_use"


def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


def route_primary_assistant(
        state: State,
) -> Literal[
    "enter_loan_tool",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == To_Loan_tool_1.__name__:
            return "enter_loan_tool"
    raise ValueError("Invalid route")

from langgraph.constants import Send
def route_to_workflow(
        state: State,
) -> Literal[
    "primary_assistant",
    "update_loan"
]:
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tool):
    return ToolNode(tool).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
