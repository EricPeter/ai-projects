from typing import Annotated, Any
from typing_extensions import TypedDict

import anthropic
from langgraph.graph import StateGraph, END

from .retrieval import hybrid_search_and_rerank


class RAGState(TypedDict):
    question: str
    documents: list[dict[str, Any]]
    answer: str
    citations: list[int]


def retrieve(state: RAGState) -> RAGState:
    docs = hybrid_search_and_rerank(state["question"])
    return {**state, "documents": docs}


def generate(state: RAGState) -> RAGState:
    client = anthropic.Anthropic()

    context_parts = []
    for i, doc in enumerate(state["documents"], 1):
        context_parts.append(f"[Source {i}, Page {doc['page_number']}]\n{doc['content']}")
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a precise Q&A assistant. Answer the question using ONLY the provided context. "
        "Cite sources by their page numbers using format (Page X). "
        "If the context does not contain the answer, say so explicitly."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer with page citations:"

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = message.content[0].text

    # Extract cited page numbers from the answer
    import re
    pages_mentioned = set(int(m) for m in re.findall(r"Page\s+(\d+)", answer, re.IGNORECASE))
    all_pages = [doc["page_number"] for doc in state["documents"]]
    citations = sorted(pages_mentioned & set(all_pages)) or sorted(set(all_pages))[:3]

    return {**state, "answer": answer, "citations": citations}


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag_graph = build_rag_graph()
