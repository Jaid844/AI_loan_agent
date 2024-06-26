from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langsmith import Client
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


dataset_name = "Loan agent response_albert_bad_profile"
# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")


def answer_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs["response"]

    # LLM grader
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}


from langsmith.evaluation import evaluate

experiment_prefix = "loan_agent_bad_profile_for_albert"
metadata = "ABC bank bad profile"
experiment_results = evaluate(
    predict_loan_agent_answer,
    data=dataset_name,
    evaluators=[answer_evaluator],
    experiment_prefix=experiment_prefix + "-response-v-reference",
    num_repetitions=1,
    metadata={"version": metadata})

