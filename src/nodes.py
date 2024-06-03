import os
from typing import Callable

from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from typing import Literal

from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import END
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, ensure_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from tools import *
from state import State
from langchain_core.pydantic_v1 import BaseModel, Field
from audio import audio_node

load_dotenv()
tools = [monthly_payment]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Loan Agent adjustment"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully calculated the loan amount ,",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },

        }


class End_of_conversation(BaseModel):
    """
    A tool to mark the end of the conversation ,if there is slight suggestion that the conversation has been concluded
    """
    dialogue: str = Field(
        description="The converstion if there is the end of the conversation"
    )

    class Config:
        schema_extra = {
            "example": {
                "dialogue": "Thanks ,bye",
            },
            "example": {
                "dialogue": "if you need anything just call us ,take care bye",
            }
        }


class To_Loan_tool_1(BaseModel):
    """
    If user agrees to pay some portion of the loan this ,This agent will help to calculate loan amount
      """
    name: str = Field(
        description="The name of the customer "
    )

    dialogue: str = Field(
        description="The converstion of the customer if he agrees to pay some portion of the loan amount"
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "jake",
                "request": "Yes I would like a loan adjustment",
            }
        }


class Assistant:

    def __init__(self, runnable: Runnable):
        self.runnable = runnable
        self.audio = audio_node()

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)  # the input are converted into dictionary key value pair

            self.audio.streamed_audio(result.content)

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
        human_messages = state['messages']
        config = ensure_config()  # Fetch from the context
        name = state['name']
        session_id = state['session_id']
        system = """You are loan agent called as Sandy from ABC bank here to discuss the loan payment this customer has 
                  a good payment history
                  This is going to be a telephonic call so play along have a small conversation
                 Your primary role is to help customer to find reason why didn't he paid the loan this month
                
                 ask him you calculate the loan amount
                   then then quietly delegate calculation of work to another agent 
                 without letting know about this agent and calculate the loan amount ,
                 this agent will calculate the loan amount for you ,you dont have to worry
                 after another agent calculate the loan amount ,
                 You just need first name to calculate the loan amount 
                 when the loan amount will be calculated the agent will tell you
                 "loan amount calculated"
                 Ask him if he agress to pay this 
                 amount if he agrees to pay that amount then tell him this will be his new amount this month
                 
                """
        human = """ \n\nHere is the user response -- {messages}
        \n\n Here is the name of the customer {name}
                 """
        primary_assitant_prompt = ChatPromptTemplate.from_messages(
            [("system", system),
             MessagesPlaceholder(variable_name="history"),
             ("human", human)]
        )
        # llm = ChatOpenAI(model='gpt-4o')
        llm = ChatOpenAI(model='gpt-3.5-turbo')
        # llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
        tools = [To_Loan_tool_1]
        primary_assitant_runnable = primary_assitant_prompt | llm.bind_tools(
            tools + [End_of_conversation]) | StrOutputParser()
        with_message_history = RunnableWithMessageHistory(
            primary_assitant_runnable,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///history_of_conversation.db"
            ),
            input_messages_key="messages",
            history_messages_key="history",
        )
        generation = with_message_history.invoke({"messages": human_messages, "name": name},
                                                 config={"configurable": {"session_id": session_id}})
        # self.audio.streamed_audio(generation)
        return {
            "messages": generation
        }

    def tool_runnable(self, state):
        session_id = state['session_id']
        human_messages = state['messages']
        name = state['name']
        llm = ChatOpenAI(model='gpt-3.5-turbo')
        # llm = ChatOpenAI(model='gpt-4o')#
        # llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
        loan_hotel_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a specialized assistant for calculating loan amount of a customer "
                 "The primary assistant delegates work to you whenever the user needs help with calculating loan amount"
                 " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                 "Once you have calculated laon amount delgate back to  main assistant."
                 " Remember that a loan amount  isn't completed until after the relevant tool has successfully been used."
                 ' then "CompleteOrEscalate" the dialog to the host assistant.'
                 " Do not waste the user's time. Do not make up invalid tools or functions."
                 "You dont need first name of the customer that's it"
                 " then end the conversation by say such word as bye "
                 "You just need first name to calculate the loan amount "
                 "Name of the customer is {name}"
                 "\n\nSome examples for which you should CompleteOrEscalate:\n"
                 " - 'Loan amount calcualted '",
                 ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "here is the human reply \n\n{messages},here is the name of the customer  {name}")
            ]
        )
        tool_1 = [monthly_payment]
        loan_tool_runnable = loan_hotel_prompt | llm.bind_tools(tool_1 + [CompleteOrEscalate]) | StrOutputParser()
        with_message_history = RunnableWithMessageHistory(
            loan_tool_runnable,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///history_of_conversation.db"
            ),
            input_messages_key="messages",
            history_messages_key="history",
        )
        generation = with_message_history.invoke({"messages": human_messages, "name": name},
                                                 config={"configurable": {"session_id": session_id}})
        #self.audio.streamed_audio(generation)
        return {
            "messages": generation,
        }

    def create_entry_node(self, assistant_name: str, new_dialog_state: str) -> Callable:
        def entry_node(state: State) -> dict:
            tool_call_id = state["messages"][-1].tool_calls[0]["id"]
            return {
                "messages": [
                    ToolMessage(
                        content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                                f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                                " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                                " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                                " Do not mention who you are - just act as the proxy for the assistant.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": new_dialog_state,
            }

        return entry_node

    def handle_tool_error(self, state) -> dict:
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

    def create_tool_node_with_fallback(self, tools: list) -> dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )


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
    safe_toolnames = [t.name for t in tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
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
        elif tool_calls[0]["name"] == End_of_conversation.__name__:
            return END
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
