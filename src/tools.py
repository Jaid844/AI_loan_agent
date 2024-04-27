import pandas as pd
from langchain_core.tools import ToolException
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.pydantic_v1 import BaseModel, Field

class CalculatorInput(BaseModel):
    name: str = Field(description="Full Name of the Person that needs to be searched ")

class tools_agent():
        def monthly_payment(self,name: str) -> int:
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


        def loan_calcualtor(self):
            monthly_calculator = StructuredTool.from_function(
                func=self.monthly_payment,
                name="Montly_Payment_Calculator",
                description="useful for when need to calculate the new amount for the loan term to be paid when name is given",
                args_schema=CalculatorInput,
                handle_tool_error=True
            )
            return monthly_calculator