import os
import time

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere import ChatCohere
from faster_whisper import WhisperModel
import sounddevice as sd
import soundfile as sf
from langchain_community.vectorstores.faiss import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException, tool
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_tool_calling_agent, \
    create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from openai import OpenAI
from audio import audio_node
from tools import tools_agent
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
# set_llm_cache(InMemoryCache())
client = OpenAI()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Loan Agent adjustment"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

embeddings = VoyageAIEmbeddings(
    model="voyage-2", batch_size=128, truncation=True
)


def loan_embeing_model():
    new_db = FAISS.load_local("faiss_index_loan_voyage1", embeddings, allow_dangerous_deserialization=True)
    new_db = new_db.as_retriever(search_kwargs={"k": 1})
    return new_db


@tool
def monthly_payment(name: str) -> int:
    """
        This tool will help to give new monthly payment for user
        :param name: Full name of the customer
        :return:
        """
    try:
        df = pd.read_csv("Loan_amount.csv")
        df.set_index('Name', inplace=True)
        interest_rate = 0.05  #
        monthly_payment = df.loc[name]['Monthly_Payment']
        new_monthly_payment = monthly_payment * (1 + interest_rate) - monthly_payment
        df.reset_index(inplace=True)
        return new_monthly_payment
    except Exception as e:
        raise ToolException("The search tool1 is not available.", e)


class Nodes():
    def __init__(self):
        self.audio = audio_node()
        self.tools = tools_agent()

    def customer_profile_summarizer(self, state):
        name = state['name']

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
            "Profile": generation,

        }

    def customer_voice_1(self, state, duration=5, fs=44100):
        print('Recording...')
        name = state['name']
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        print('Recording complete.')
        filename = 'myrecording.wav'
        sf.write(filename, myrecording, fs)
        audio_file = open(filename, "rb")
        # model = WhisperModel('base.en', device='cpu', compute_type="int8")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        # for segment in segments:
        #    transcription=segment.text
        return {

            "transcription": transcription.text,
            "name": name,

        }

    def customer_voice_2(self, state, duration=5, fs=44100):
        print('Recording...')
        name = state['name']
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        print('Recording complete.')
        filename = 'myrecording.wav'
        sf.write(filename, myrecording, fs)
        audio_file = open(filename, "rb")
        # model = WhisperModel('base.en', device='cpu', compute_type="int8")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        # for segment in segments:
        #    transcription=segment.text
        return {

            "transcription": transcription.text,
            "name": name
        }

    def Good_Profile_Chain(self, state):
        Profile = state['Profile']
        session_id = state['session_id']
        transcription = state['transcription']
        adjustment = state['adjustment']
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        example = [
            {
                "Loan Agent (Sandy)": " Good morning/afternoon, [Customer's Name]. This is Sandy calling from ABC bank. I hope you're doing well today.?",
                "Customer": "Good morning/afternoon. Yes, thank you, I'm doing fine. How can I assist you?"},

            {
                "Loan Agent (Sandy)": "  I'm calling today to discuss your recent loan payment. I noticed there's been a delay, which is unusual given your excellent payment history. I wanted to check in with you to ensure everything is alright on your end.",
                "Customer": " I appreciate your concern. Unfortunately, I encountered an unexpected issue with my finances this month that caused the delay in payment."},

            {
                "Loan Agent (Sandy)": ": I understand, [Customer's Name]. Life can be unpredictable, and these things happen. Your consistent payment history hasn't gone unnoticed, and I'm here to assist you in any way I can. Would you like to discuss your situation further so we can find a suitable solution together?",
                "Customer": " Yes, please. I would appreciate any assistance you can offer."},

            {
                "Loan Agent (Sandy)": "Certainly. Let's review your current situation and explore some options to help you get back on track. We could consider adjusting your payment schedule, setting up a payment plan, or exploring other alternatives that best fit your circumstances. Does that sound like a good starting point for us?",
                "Customer": " Yes, that sounds reasonable."},
            {
                "Loan Agent (Sandy)": " Firstly, let's review your current financial situation together. This will help us understand the extent of the issue and determine the best course of action. Do you have a clear picture of your expenses and income for the upcoming months?.",
                "Customer": "Yes, I have some rough estimates."},
            {
                "Loan Agent (Sandy)": "Excellent. Let's start by identifying any discretionary expenses that could be reduced or eliminated temporarily to free up funds for your loan payments. Additionally, if you have any assets or savings that could be used to cover the outstanding amount, now might be the time to consider utilizing them.",
                "Customer": " That makes sense. I'll take a closer look at my budget and see where I can make adjustments."},

            {
                "Loan Agent (Sandy)": "Perfect. Once you've identified potential areas for savings, we can discuss restructuring your payment plan. This could involve extending the loan term, adjusting the monthly installments, or exploring alternative payment arrangements that better align with your current financial situation.",
                "Customer": " Okay, I'll gather all the necessary information and get back to you with my proposed plan.."},

            {
                "Loan Agent (Sandy)": "That sounds like a plan. In the meantime, if you have any questions or need further assistance, don't hesitate to reach out to me. I'm here to support you every step of the way.",
                "Customer": " Thank you so much for your help. I feel more confident about resolving this issue now."},

            {
                "Loan Agent (Sandy)": "You're very welcome, [Customer's Name]. Remember, we're a team, and together we'll find a solution that works for you. Take your time, and when you're ready, we'll discuss your proposed plan in detail.",
                "Customer": "  I appreciate your support. I'll be in touch soon."},

            {"Loan Agent (Sandy)": "Sounds good. Take care, and have a great day!",
             "Customer": " You too. Goodbye"},

        ]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", "{Loan Agent (Sandy)}"),
                ("human", "{Customer}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=prompt,
            examples=example,
        )
        system = """
               You are loan agent called as Sandy from ABC bank here to disscuss the loan payment this customer has a
                good payment history 
               ,Might have some financal issue can you ask this person ,why didn't he paid this month,
               urge him if he can pay some portion of the loan this month ,tell the customer you(Sandy) are ready to
                adjust the loan amount 
               for him due to his good credit history
               First ask him if he ready for adjustment in his loan term ,then use adjust the loan amount 
               The loan adjustment will be calculated at mean time so wait unitl this is calculated engage the user in 
               conversation
               
               \nMake sure you dont repeat yourself during the conversation.\n
               \nMake the conversation in telephonic ,make it small and sweet\n
               """
        human = """
                   Here is the customer profile {profile} \n\nHere is the user response {user-query}
                   \n\n Here is what he/she needs to pay after adjustment (this might not be calculated yet)\n\n
                   {adjustment}                  
                   """

        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="history"),
                ("human", human),
            ]
        )
        rag_chain = final_prompt | llm | StrOutputParser()
        with_message_history = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///history_of_conversation.db"
            ),
            input_messages_key="user-query",
            history_messages_key="history",
        )

        generation = with_message_history.invoke({"profile": Profile, "user-query": transcription,
                                                  "adjustment": adjustment},
                                                 config={"configurable": {"session_id": session_id}})
        self.audio.streamed_audio(generation)
        return {
            "generation": generation,
        }

    def Bad_Profile_Chain(self, state):
        profile = state['Profile']
        session_id = state['session_id']
        transcription = state['transcription']
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        system = """You are loan agent called as Sandy from ABC bank here to disscuss the loan payment this customer has a bad payment history of payments
        ,Might have some financal issue can you ask this person ,why didnt he paid this month,
         can he pay some amount if the conversation is not good ,give him the warning the bank might take some legal action against him
        This is a telephonic call so make a call ,talk in a that manner in small and precise manner
        After making the call/concluding the conversation just say Goodbye """
        human = """Make sure you dont repeat yourself during the conversation.
                    Here is the customer profile {profile} \n\n Here is the user response \n\n ---{user-query}
                
                    
                    """
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="history"),
                ("human", human),
            ]
        )
        rag_chain = final_prompt | llm | StrOutputParser()
        with_message_history = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///history_of_conversation.db"
            ),
            input_messages_key="user-query",
            history_messages_key="history",
        )
        generation = with_message_history.invoke({"profile": profile, "user-query": transcription,
                                                  },
                                                 config={"configurable": {"session_id": session_id}})

        self.audio.streamed_audio(generation)
        return {
            "generation": generation,
        }

    def grade_conversation(self, state):
        class GradeConclusion(BaseModel):
            """Binary score for conversation to see if the conversation has been reached in conclusion
            typically indicates that conversation have been completed  ."""

            binary_score: str = Field(description="Conversation has been reached to conclusion/completed 'yes' or 'no'")

        generation = state['generation']
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeConclusion)
        system = """As a grader assessing the conversation between the user and AI, your task is to determine if the conversation contains keywords or semantic cues that signal the conclusion of the interaction, such as "Bye." Grade it as relevant if such indicators are present.
        Provide a binary score of 'yes' or 'no' to indicate whether the conversation has concluded. 'yes' means the conversation has ended, and 'no' means it is still ongoing.
                    """
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: {conversation}"),
            ]
        )
        retrieval_grader = grade_prompt | structured_llm_grader
        score = retrieval_grader.invoke({"conversation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("--CONVERSATION ENDED")
            return "END"
        else:
            print("--Conversation CONTINUES")
            return "customer_voice"

    def grade_profile(self, state):
        print("----CHECKING THE IF THE PROFILE IS GOOD OR BAD")
        Profile = state['Profile']

        class GradeConclusion(BaseModel):
            """Binary score for profile to see if the profile is good profile or the bad profile
            """

            binary_score: str = Field(
                description="Profile if they are good or bad based on credit history, 'Good' or 'Bad'")

        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
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

        score = customer_profile_grader.invoke({"profile": Profile})
        if score.binary_score == "Good":
            print("--FORKING TO GOOD  PROFILE CHAIN")
            return "Good_Profile_Voice"
        else:
            print("--FORKING TO POOR  PROFILE CHAIN")
            return "Bad_Profile_Voice"

    def grade_loan_adjustment(self, state):
        print("CHECKING IF THE LOAN ADJUSTMENT HAS BEEN DONE")
        ai_voice = state['generation']

        class Grade_loan_modification(BaseModel):
            """Binary score for conversation to see if loan adjustment has been done or not
          """

            binary_score: str = Field(
                description="Conversation has been reached to Loan modification terms 'yes' or 'no'")

        system = """ As a grader assessing the conversation between the user and AI,your task is to determine if the conversation contains
      keyword or semantic cues that signals any loan modification verdict such 'We will be adjusting your loan terms' .Grade it as relevant if such indicators are present.
      Provide a binary score of 'yes' or 'no' to indicate whether the conversation has loan modification term in it 
      'yes' means the conversation has reached to loan modification , and 'no' means it has no
      """
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm_grader = llm.with_structured_output(Grade_loan_modification)
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: {conversation}"),
            ]
        )
        retrieval_grader_1 = grade_prompt | structured_llm_grader
        score = retrieval_grader_1.invoke({"conversation": ai_voice})
        grade = score.binary_score
        if grade == "yes":
            print("--CONVERSATION ROUTED TO LOAN ADJUSTMENT")
            return "Loan_Adjustment_Agent"
        else:
            print("--LOAN ADJUSTMENT HAS NOT BEEN DONE")
            return "customer_voice"

    def loan_adjustment_agent(self, state):
        generation = state['generation']
        system = """
        You are loan agent that helps people calculate their monthly payment with the available tool you calculate them their annual loan 
        Your final answer should include what their loan amount they will pay this month

        """
        human = '''TOOLS
        ------
        Assistant can ask the user to use tools to look up information that may be helpful in             answering the users original question. The tools the human can use are:

        {tools}

        RESPONSE FORMAT INSTRUCTIONS
        ----------------------------

        When responding to me, please output a response in one of two formats:

        **Option 1:**
        Use this if you want the human to use a tool./

        Markdown code snippet formatted in the following schema:

        ```json
        {{
            "action": string, \ The action to take. Must be one of {tool_names}
            "action_input": string \ The input to the action
        }}
        ```

        **Option #2:**
        Use this if you want to respond directly to the human. Markdown code snippet formatted             in the following schema:

        ```json
        {{
            "action": "Final Answer",
            "action_input": string \ You should put what you want to return to use here
        }}
        ```

        USER'S INPUT
        --------------------
        Here is the user's input (remember to respond with a markdown code snippet of a json             blob with a single action, and NOTHING else):

        {input}'''
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", human),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        llm = ChatOpenAI(model='gpt-3.5-turbo')
        tools = [monthly_payment]
        agent = create_json_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )
        generation = agent_executor.invoke({"input": generation})
        return {
            "adjustment": generation['output']
        }
