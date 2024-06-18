import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS

from langchain_core.tools import ToolException
from langchain.tools import tool

from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

embeddings = VoyageAIEmbeddings(
    model="voyage-2", batch_size=128, truncation=True
)


@tool
def monthly_payment(name: str, rate: int) -> str:
    """
            This tool will help to give new monthly payment for user
            :param rate:  rate at which loan amount will be calculated
            :param name:  name of the customer
            :return: string the amount the customer will pay this month
            """
    try:
        df = pd.read_csv("Loan_amount.csv")
        df.set_index('Name', inplace=True)
        interest_rate = rate/100  #
        monthly_payment = df.loc[name]['Monthly_Payment']
        new_monthly_payment = monthly_payment * (1 + interest_rate) - monthly_payment
        df.reset_index(inplace=True)
        if rate == 10:
            return f"This will be the last {new_monthly_payment}  payment for the customer {name}"
        elif rate == 5:
            return f"The initial discounted loan amount will be {new_monthly_payment} for the customer {name}"
    except Exception as e:
        raise ToolException("The search tool1 is not available.", e)


def loan_embeing_model():
    new_db = FAISS.load_local("faiss_index_loan_voyage1", embeddings, allow_dangerous_deserialization=True)
    new_db = new_db.as_retriever(search_kwargs={"k": 1})
    return new_db




def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
