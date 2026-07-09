from typing import TypedDict
from langgraph.graph import StateGraph, END
from model import ask_llm

class AgentState(TypedDict):
    question: str
    answer: str


def chatbot_node(state: AgentState):

    answer = ask_llm(state["question"])

    return {
        "question": state["question"],
        "answer": answer
    }


graph = StateGraph(AgentState)

graph.add_node("chatbot", chatbot_node)

graph.set_entry_point("chatbot")

graph.add_edge("chatbot", END)

agent = graph.compile()