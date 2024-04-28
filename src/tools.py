import pandas as pd
from langchain_core.tools import ToolException
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.pydantic_v1 import BaseModel, Field

class CalculatorInput(BaseModel):
    name: str = Field(description="Full Name of the Person that needs to be searched ")

class tools_agent():
        @tool
        def monthly_payment(self,name: str) -> int:
            """
            This tool will help to give new monthly payment for user
            :param name: Full name of the customer
            :return:
            """
            try:
                df =pd.read_csv("Loan_amount.csv")
                df.set_index('Name', inplace=True)
                interest_rate = 0.05  #
                monthly_payment =df.loc[name]['Monthly_Payment']
                new_monthly_payment = monthly_payment * (1 + interest_rate ) -monthly_payment
                df.reset_index(inplace=True)
                return new_monthly_payment
            except Exception as e:
                raise ToolException("The search tool1 is not available." ,e)

