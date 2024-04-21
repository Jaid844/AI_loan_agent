import os
import time
from langchain_core.prompts import ChatPromptTemplate
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from openai import OpenAI
from audio import audio_node
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

    def customer_voice(self,state,duration=8, fs=44100):
        print('Recording...')
        name = state['name']
        session_id=state['session_id']
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
            "session_id":session_id
        }

    def ai_voice(self,state):
        transcription = state['transcription']
        session_id =state['session_id']
        name = state['name']
        #llm = ChatGroq(model="gemma-7b-it", temperature=0.7)
        llm = ChatOpenAI(model="gpt-3.5-turbo",  temperature=0)
        #llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85,convert_system_message_to_human=True)
        #llm = ChatCohere(max_tokens=30)
        documents = loan_embeing_model().get_relevant_documents(name)
        documents = [d.page_content for d in documents]
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
        system = """You are a loan agent calling a customer .
The customer has a history of either good or bad payment habits on their loan.
Your goal is to:
Address the recent payment delay in a tactful manner.
Understand the reason behind the delay (if applicable, for bad payment history).
Offer assistance to resolve the current delay.
Emphasize the importance of on-time payments for a healthy financial relationship.
Maintain a professional and courteous tone throughout the call.
Conversation Flow:

Introduction:

You: "Hello, this is Sandy calling from ABC BANK. May I speak to [Customer Name]?"
Payment Delay:

Good Payment History:
You: "Hi [Customer Name], I'm calling to follow up on a recent payment for your loan. It appears there may have been a slight delay this time."
Bad Payment History:
You: "Hi [Customer Name], I'm calling to discuss your loan payment. We noticed a delay for this month's payment."
Understanding the Reason (if applicable):

Bad Payment History:
You: "Is everything alright? Would you like to discuss any challenges you might be facing with making the payment?" (Avoid mentioning past delays directly.)
Offering Assistance:

You: "We're here to help! We can offer flexible options to get your account current. Would you be interested in discussing those?"
Importance of On-Time Payments:

You: "Maintaining timely payments is crucial for a positive credit history and a healthy financial relationship. We're here to support you."
Conclusion:

Based on the customer's response, offer additional solutions or next steps.
End with a courteous closing: "Thank you for your time, [Customer Name]. Have a great day!" or "Goodbye".

Remember:
You have history of the conversation between customer and you ,so dont try to repeat the conversation
Adapt your approach based on the customer's payment history.
Don't repeat information or the customer's name excessively.
Maintain a professional and helpful tone throughout the call.
            """
        human = """Make sure you dont repeat yourself during the conversation.
        Here is the customer profile {customer} \n\n User  query {userquery}"""
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="history"),
                few_shot_prompt,
                ("human", human),
            ]
        )
        rag_chain = final_prompt | llm | StrOutputParser()
        with_message_history = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id, connection_string="sqlite:///history_of_conversation.db"
            ),
            input_messages_key="userquery",
            history_messages_key="history",
        )
        generation = with_message_history.invoke(
            {"userquery":transcription , "customer": documents},
            config={"configurable": {"session_id": session_id}}
        )

        self.audio.streamed_audio(generation)    #Voice out TTS model from OpenAI
        return {

            "generation": generation,
            "name": name,
            "session_id":session_id
        }




    def grade_conversation(self,state):
        class GradeConclusion(BaseModel):
            """Binary score for conversation to see if the conversation has been reached in conclusion
            typically indicates that conversation have been completed  ."""

            binary_score: str = Field(description="Conversation has been reached to conclusion/completed 'yes' or 'no'")
        preamble = """As a grader assessing the conversation between the user and AI, your task is to determine if the conversation contains keyword(s) or semantic meaning indicative of concluding the interaction, such as "Bye."
            Grade it as relevant if such indicators are present. 
            Provide a binary score of 'yes' or 'no' to indicate whether the conversation has been completed or concluded.
            'yes' means the conversation has been ended or concluded 
            'no '  means the conversation is still going on
            """
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