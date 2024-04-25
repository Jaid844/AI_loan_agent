import os
import time
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
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
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import ToolException
from langchain.agents import AgentExecutor,  create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate
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
#set_llm_cache(InMemoryCache())
client = OpenAI()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Loan Agent"
os.environ['KMP_DUPLICATE_LIB_OK']='True'


embeddings = VoyageAIEmbeddings(
     model="voyage-2",batch_size=128,truncation=True
)
def loan_embeing_model():
    new_db = FAISS.load_local("faiss_index_loan_voyage1", embeddings,allow_dangerous_deserialization=True)
    new_db=new_db.as_retriever(search_kwargs={"k": 1})
    return new_db

class Nodes():
    def __init__(self):
        self.audio=audio_node()
        self.tools=tools_agent()

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

    def customer_voice_1(self,state,duration=5, fs=44100):
        print('Recording...')
        name = state['name']
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        print('Recording complete.')
        filename = 'myrecording.wav'
        sf.write(filename, myrecording, fs)
        audio_file = open(filename ,"rb")
       # model = WhisperModel('base.en', device='cpu', compute_type="int8")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        #for segment in segments:
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
        transcription = state['transcription']
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        system = '''You are loan agent called as Sandy from ABC bank here to disscuss the loan payment this customer has a good payment history 
                ,Might have some financal issue can you ask this person ,why didnt he paid this month,
        Use the tool given to ask the user 
        pay some portion of the amount ,always use this tool when you  need to give him new monthly payment,
        You have access of this following tool
        and After making the call/concluding the conversation just say Goodbye 

         :

        {tools}

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
          "action": $TOOL_NAME,
          "action_input": $INPUT
        }}
        ```

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{
          "action": "Final Answer",
          "action_input": "Final response to human"
        }}

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools only when a valid  person  name is given. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

        human = '''{userquery}

        {agent_scratchpad}
        Here is the user profile \n{profile}
        (reminder to respond in a JSON blob no matter what)'''

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        tools=[self.tools.loan_calcualtor()]
        agent =create_structured_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        generation = agent_executor.invoke({"profile": Profile, "userquery": transcription})
        self.audio.streamed_audio(generation)
        return {
            "generation": generation,
        }

    def Bad_Profile_Chain(self, state):
        profile = state['Profile']
        transcription = state['transcription']
        llm = ChatGroq(model="llama3-8b-8192", temperature=0)
        system = '''You are loan agent called as Sandy from ABC bank here to disscuss the loan payment this customer has a good payment history 
                        ,Might have some financal issue can you ask this person ,why didnt he paid this month,
                Use the tool given to ask the user 
                pay some portion of the amount ,always use this tool when you  need to give him new monthly payment,
                You have access of this following tool
                Might have some financal issue can you ask this person ,why didnt he paid this month,
         can he pay some amount if the conversation is not good ,give him the warning the bank might take some legal action against him
        This is a telephonic call so make a call ,talk in a that manner in small and precise manner
        After making the call/concluding the conversation just say Goodbye 
                 :

                {tools}

                Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

                Valid "action" values: "Final Answer" or {tool_names}

                Provide only ONE action per $JSON_BLOB, as shown:

                ```
                {{
                  "action": $TOOL_NAME,
                  "action_input": $INPUT
                }}
                ```

                Follow this format:

                Question: input question to answer
                Thought: consider previous and subsequent steps
                Action:
                ```
                $JSON_BLOB
                ```
                Observation: action result
                ... (repeat Thought/Action/Observation N times)
                Thought: I know what to respond
                Action:
                ```
                {{
                  "action": "Final Answer",
                  "action_input": "Final response to human"
                }}

                Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools only when a valid  person  name is given. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

        human = '''{userquery}

                {agent_scratchpad}
                Here is the user profile \n{profile}
                (reminder to respond in a JSON blob no matter what)'''
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        tools = [self.tools.loan_calcualtor()]
        agent = create_structured_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        generation = agent_executor.invoke({"profile": profile, "userquery": transcription})

        self.audio.streamed_audio(generation)
        return {
            "generation": generation,
        }

    def grade_conversation(self,state):
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
        retrieval_grader= grade_prompt | structured_llm_grader
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