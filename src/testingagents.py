from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langsmith import Client
from langsmith.schemas import Run, Example

from main import WorkFlow
from langchain import hub
from langchain_openai import ChatOpenAI

app = WorkFlow().app
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": "2-ALB-e",

    }
}
client = Client()


def predict_loan_agent_answer(example: dict):
    """Use this for answer evaluation"""
    msg = {"messages": ("user", example["input"]), "name": "Albert Einstein"}
    messages = app.invoke(msg, config)
    return {"response": messages['messages'][-1].content}


dataset_name = "Loan agent response_albert"
# Grade prompt

def find_tool_calls(messages):
    """
    Find all tool calls in the messages returned
    """
    tool_calls = [tc['name'] for m in messages['messages'] for tc in getattr(m, 'tool_calls', [])]
    return tool_calls
def contains_all_tool_calls_any_order(root_run: Run, example: Example) -> dict:
    """
    Check if all expected tools are called in any order.
    """
    expected = ['sql_db_list_tables', 'sql_db_schema', 'sql_db_query_checker', 'sql_db_query', 'check_result']
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(messages)
    # Optionally, log the tool calls -
    print("Here are my tool calls:")
    print(tool_calls)
    if set(expected) <= set(tool_calls):
        score = 1
    else:
        score = 0
    return {"score": int(score), "key": "multi_tool_call_any_order"}




from langsmith.evaluation import evaluate

experiment_prefix = "loan_agent_bad_profile_for_albert"
metadata = "ABC bank bad profile"
experiment_results = evaluate(
    predict_loan_agent_answer,
    data=dataset_name,
    evaluators=[contains_all_tool_calls_any_order],
    experiment_prefix=experiment_prefix + "-trajectory",
    num_repetitions=3,
    metadata={"version": metadata},
)

