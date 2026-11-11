"""A real LangGraph router: input -> intent -> retrieval -> specialist -> formatter."""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from tools.study_tools import retrieve_relevant_chunks
from agents.study_agent import generate

class StudyState(TypedDict, total=False):
    query: str; intent: Literal["qa", "summary", "mcq", "exam"]
    store: object; context: str; answer: str; sources: list[str]; count: int; detail: str

def classify(state: StudyState):
    q = state["query"].lower()
    intent = "mcq" if any(x in q for x in ["mcq", "quiz", "multiple choice"]) else "exam" if any(x in q for x in ["exam", "2-mark", "5-mark", "10-mark"]) else "summary" if any(x in q for x in ["summar", "notes", "chapter"]) else "qa"
    return {"intent": intent}

def retrieve(state: StudyState):
    hits = retrieve_relevant_chunks(state["store"], state["query"])
    return {"context": "\n\n".join(f"[{c.source}, p.{c.page}]\n{c.text}" for c, _ in hits), "sources": list(dict.fromkeys(f"{c.source}, p.{c.page}" for c, _ in hits))}

def answer(state): return {"answer": generate(f"Answer this question: {state['query']}. Cite page labels like [file, p.X] for claims.", state["context"])}
def summary(state): return {"answer": generate(f"Make {state.get('detail','short')} structured notes about: {state['query']}. Include definitions and key takeaways.", state["context"])}
def mcq(state): return {"answer": generate(f"Create exactly {state.get('count',5)} MCQs about: {state['query']}. Four options A-D, answer, and explanation each.", state["context"])}
def exam(state): return {"answer": generate(f"Generate 3 two-mark, 3 five-mark, and 2 ten-mark questions about: {state['query']}, with quick revision notes.", state["context"])}
def format_response(state): return {"answer": state["answer"] + "\n\n---\n**Retrieved sources:** " + "; ".join(state.get("sources", []))}

def route(state: StudyState): return state["intent"]

def build_workflow():
    graph = StateGraph(StudyState)
    graph.add_node("user_input", lambda state: state); graph.add_node("intent_classification", classify)
    graph.add_node("document_retrieval", retrieve); graph.add_node("question_answering", answer)
    graph.add_node("summary_generation", summary); graph.add_node("mcq_generation", mcq)
    graph.add_node("exam_generation", exam); graph.add_node("response_formatting", format_response)
    graph.set_entry_point("user_input"); graph.add_edge("user_input", "intent_classification")
    graph.add_edge("intent_classification", "document_retrieval")
    graph.add_conditional_edges("document_retrieval", route, {"qa":"question_answering", "summary":"summary_generation", "mcq":"mcq_generation", "exam":"exam_generation"})
    for node in ["question_answering", "summary_generation", "mcq_generation", "exam_generation"]: graph.add_edge(node, "response_formatting")
    graph.add_edge("response_formatting", END)
    return graph.compile()
