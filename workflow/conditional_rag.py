import os
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

embeddings = MistralAIEmbeddings()

def build_retriever(path:str):
    load = PyPDFLoader(path)
    doc = load.load()

    splitter = RecursiveCharacterTextSplitter(chunk = 800,chunk_overlap=400)

    chunks = splitter.split_documents(doc)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

acedemic_retriever = build_retriever("academics_handbook.pdf")
fee_retriever = build_retriever("fee_structure.pdf")

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.4)

class AnalyserState(TypedDict):
    programme : str
    messages : Annotated[list[dict],add_messages]
    query : str
    retrived_context : str

def classifier_node(state: AnalyserState) -> dict:
    """Look at the latest user message and decide which path to take"""

    last_msg = state["messages"][-1]["content"]

    prompt = (
        f"""Classify the following student query into exactly one category:

        'academic', 'fee', or 'general'.

        Use 'academic' for questions about attendance, exams, grading, credits,
        promotion, course structure, summer training, or degree requirements.

        Use 'fee' for questions about tuition, payment, refund, late charges,
        scholarships, or any money-related topic.

        Use 'general' for greetings, casual talk, or anything not related to
        the college rules or fee.

        Query: {last_msg}

        Return only one word: academic, fee, or general."""
    )

    response = llm.invoke(prompt)
    catagory = response.content.strip().lower()

    if "academic" in catagory:
        category = "academic"
    elif "fee" in catagory:
        category = "fee"
    else:
        category = "general"

    return {"query" : category}

def academic_rag_node(state: AnalyserState) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: AnalyserState) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: AnalyserState) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state: AnalyserState) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


#step 4 - router function 

def route_query(state:AnalyserState):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


#step 5 - Building the graph 

graph = StateGraph(AnalyserState)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)

#edges 

graph.add_edge(START,"classifier")

graph.add_conditional_edges(
    "classifier",route_query
)

graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response",END)

app = graph.compile()
#step 6 - Run the code 

print("welcome to the College assistant \n\n")

print("which programe are you in ")
print("1. BCA")
print("2. BBA")
print("3. B.com (H)")

choice = input("\nEnter 1, 2 or 3 ")

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}
student_programme = programme_map.get(choice)

print(f"\nGreat! You're set as a {student_programme} student.")

while True:
    user_query = input("You:  ")

    if user_query.lower() in ["exit","quit"]:
        break
    
    result = app.invoke({
        "programme": student_programme,
        "messages": [("human",user_query)]
    })

    print(f"Assistant : {result['messages'][-1].content}")