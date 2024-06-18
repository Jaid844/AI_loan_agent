import os
from typing import Callable, Union

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langgraph.prebuilt import ToolExecutor
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from typing import Literal

from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import END
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, \
    FewShotChatMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, ensure_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from tools import *
from state import State
from langchain_core.pydantic_v1 import BaseModel, Field
from audio import audio_node

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
            "example": {
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "reason": """The calculated loan amount for this month, after applying a 5% discount, is $16.67. 
Would you like to proceed with this amount, James?""",
            },
            "example 3": {
                "reason": "I need to search the user's emails or calendar for more information.",
            },
            "example 4": {

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
    dialogue: str = Field(
        description="The conversion of the customer if he agrees to pay some portion of the loan amount"
    )


class To_Loan_tool_1(BaseModel):
    """
    If user agrees to pay some portion of the loan this ,This agent will help to calculate loan amount
    If user feels difficult with the presented loan amount ,this function still can be used to calculate
    the new loan amount with new rate

      """

    rate: Union[loan_amount_5, loan_amount_10] = Field(discriminator="rate")

    class Config:
        schema_extra = {
            "example 1": {
                "rate": 5,
                "name": "jake",
                "dialogue": "Yes I would like a loan adjustment",
            },
            "example 2": {
                "rate": 10,
                "name": "shela",
                "dialogue": "The loan adjusment is too steep for me",
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
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class Nodes():
    def __init__(self):
        self.audio = audio_node()

    def customer_profile_summarizer(self, state):
        config = ensure_config()  # Fetch from the context
        configuration = config.get("configurable", {})
        name = configuration.get("name", None)
        documents = loan_embeing_model().get_relevant_documents(name)
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
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
        llm = ChatOpenAI(model='gpt-3.5-turbo')
        messages = state['messages']
        name = state['name']
        session_id = state['session_id']
        system = """"You are loan agent called as Sandy from ABC bank here to discuss the loan payment this customer has 
                  a good payment history
                  This is going to be a telephonic call so play along have a small conversation
                  The reason for this call is to find why didn't he paid this month portion
                  ,and you will calculate the loan amount
                 Your primary role is to help customer to find reason why didn't he paid the loan this month                
                 You will try to do loan adjustments ,for that you only need the first name,
                You cannot calculate the loan adjustment ,there is separate assistant that will be able to calculate the
                loan adjustment for you,delegate this task to another agent quietly,The user does not need to know    
                Remember to tell the user they will get 5% discount in their loan amount ,if they agree then that will
                be their loan amount ,if they disagree offer them 10 % discount in their loan amount only if they 
                disagree in 5 % loan amount calculation
                
                Here is the name of the customer {name}                 
                You are given set of example so you can reference from it
                """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system}"),
                ("human", "{placeholder}"),
            ]
        )
        examples = [
            {
                "system": "Good morning/afternoon, [Customer's Name]. This is Sandy calling from ABC bank. I hope you're doing well today.",
                "placeholder": "Good morning/afternoon. Yes, thank you, I'm doing fine. How can I assist you?"
            },
            {
                "system": "I'm calling today to discuss your recent loan payment. I noticed there's been a delay, which is unusual given your excellent payment history. I wanted to check in with you to ensure everything is alright on your end.",
                "placeholder": "I appreciate your concern. Unfortunately, I encountered an unexpected issue with my finances this month that caused the delay in payment."
            },
            {
                "system": "I understand, [Customer's Name]. Life can be unpredictable, and these things happen. Your consistent payment history hasn't gone unnoticed, and I'm here to assist you in any way I can. Would you like to discuss your situation further so we can find a suitable solution together?",
                "placeholder": "Yes, please. I would appreciate any assistance you can offer."
            },
            {
                "system": "Certainly. Let's review your current situation and explore some options to help you get back on track. We could consider adjusting your payment schedule, setting up a payment plan, or exploring other alternatives that best fit your circumstances. Does that sound like a good starting point for us?",
                "placeholder": "Yes, that sounds reasonable."
            },
            {
                "system": "Firstly, let's review your current financial situation together. This will help us understand the extent of the issue and determine the best course of action. Do you have a clear picture of your expenses and income for the upcoming months?",
                "placeholder": "Yes, I have some rough estimates."
            },
            {
                "system": "Excellent. Let's start by identifying any discretionary expenses that could be reduced or eliminated temporarily to free up funds for your loan payments. Additionally, if you have any assets or savings that could be used to cover the outstanding amount, now might be the time to consider utilizing them.",
                "placeholder": "That makes sense. I'll take a closer look at my budget and see where I can make adjustments."
            },
            {
                "system": "Perfect. Once you've identified potential areas for savings, we can discuss restructuring your payment plan. This could involve extending the loan term, adjusting the monthly installments, or exploring alternative payment arrangements that better align with your current financial situation.",
                "placeholder": "Okay, I'll gather all the necessary information and get back to you with my proposed plan."
            },
            {
                "system": "That sounds like a plan. In the meantime, if you have any questions or need further assistance, don't hesitate to reach out to me. I'm here to support you every step of the way.",
                "placeholder": "Thank you so much for your help. I feel more confident about resolving this issue now."
            },
            {
                "system": "You're very welcome, [Customer's Name]. Remember, we're a team, and together we'll find a solution that works for you. Take your time, and when you're ready, we'll discuss your proposed plan in detail.",
                "placeholder": "I appreciate your support. I'll be in touch soon."
            },
            {
                "system": "Before we end this call, I'd like to offer an additional option that might help. Given your excellent payment history, we can offer you a 10% discount on your monthly payments. This would reduce your payment from $400 to $360.",
                "placeholder": "Really? That would be incredibly helpful. Thank you!"
            },
            {
                "system": "You're welcome, [Customer's Name]. We'll send you the updated payment schedule and the details of your new monthly payment terms. If you have any further questions, feel free to contact me.",
                "placeholder": "Thank you, Sandy. I'll look out for the updated information."
            },
            {
                "system": "My pleasure. Have a great day, [Customer's Name].",
                "placeholder": "You too. Goodbye."
            }
        ]

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            input_variables=["messages"],
            examples=examples,
            example_prompt=prompt
        )
        primary_assistant_prompt = ChatPromptTemplate.from_messages(
            [("system", system),
             few_shot_prompt,
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
        return {
            "messages": generation
        }

    def tool_runnable(self):
        llm = ChatOpenAI(model='gpt-4-1106-preview')
        system = """You are a specialized assistant for calculating loan amount of a customer 
                 The primary assistant delegates work to you whenever the user needs help with calculating loan amount
                  When searching, be persistent. Expand your query bounds if the first search returns no results. 
                 Once you have calculated loan amount delegate back to  main assistant.
                  Remember that a loan amount  isn't completed until after the relevant tool has successfully been used.
                  then "CompleteOrEscalate" the dialog to the host assistant.
                  Do not waste the user's time. Do not make up invalid tools or functions.
                 You dont need first name of the customer that's it
                  then end the conversation by say such word as bye 
                 You just need first name to calculate the loan amount 
                 Remember to tell the user they will get 5% discount in their loan amount ,if they agree then that will
                be their loan amount ,if they disagree offer them 10 % discount in their loan amount only if they 
                disagree in 5 % loan amount calculation
                 Name of the customer is {name}
                 The loan tool will tell how much amount will the customer will pay this month
                 If you have calculated the loan amount then use  CompleteOrEscalate
                 \n\nSome examples for which you should CompleteOrEscalate:\n"
                  - 'Loan amount calculated ',
                  -The loan amount for this month will be $16.67 for {name}
                 tell him this will be the loan amount the user have pay for this month"""

        # llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        loan_hotel_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
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
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
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
