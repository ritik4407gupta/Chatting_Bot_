import os
from typing import TypedDict,Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)

def merge_score_dicts(existing : dict , newupdate : dict)->dict:
    if existing is None:
        return newupdate
    return {**existing,**newupdate}

#state
class AnalyserState(TypedDict):
    raw_data : str
    safety_score : Annotated[dict[str,int],merge_score_dicts]

#nodes
def toxicity_node(state: AnalyserState) -> dict:
    print("\n\n🛡️ [Branch 1] Analyzing Toxicity & Hate Speech...")

    prompt = f"""
    Analyze the text for toxicity, hate speech, harassment, and offensive language.
    Return only one integer from 0 (safe) to 100 (highly toxic).
    Text: {state["raw_data"]}
    Output only the number.
    """

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"toxicity_level": score}}

def copyright_node(state: AnalyserState) -> dict:
    print("\n\n📚 [Branch 2] Checking Copyright Risk...")

    prompt = f"""
    Analyze the text for possible copyright infringement or copied content.
    Return only one integer from 0 (no risk) to 100 (high risk).
    Text: {state["raw_data"]}
    Output only the number.
    """

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"copyright_risk": score}}

def culture_node(state: AnalyserState) -> dict:
    print("\n\n🌍 [Branch 3] Evaluating Cultural Sensitivity...")

    prompt = f"""
    Analyze the text for cultural, religious, or ethnic insensitivity.
    Return only one integer from 0 (respectful) to 100 (highly offensive).
    Text: {state["raw_data"]}
    Output only the number.
    """

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"cultural_sensitivity": score}}

graph = StateGraph(AnalyserState)

graph.add_node("toxicity",toxicity_node)
graph.add_node("copyright",copyright_node)
graph.add_node("culture",culture_node)

graph.add_edge(START,"toxicity")
graph.add_edge(START,"copyright")
graph.add_edge(START,"culture")

graph.add_edge("toxicity",END)
graph.add_edge("copyright",END)
graph.add_edge("culture",END)

app = graph.compile()

sample_script = """
    I copied several paragraphs from a famous novel and used them in my blog without giving credit. 
    Also, people from that particular community are lazy, dishonest, and cannot be trusted. 
    I hate interacting with them because they always create problems. Everyone should avoid them and never rely on their opinions or work.
"""

initial = {
    "raw_data": sample_script,
    "safety_score": {}
}

response = app.invoke(initial)

print("\n\n",response['safety_score'])