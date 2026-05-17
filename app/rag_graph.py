from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

from app import config
from app.ingestion import load_vector_store


class GraphState(TypedDict, total=False):
    question: str
    current_query: str
    top_k: int
    retry_count: int
    retrieved_docs: list[Document]
    relevant_docs: list[Document]
    web_results: list[Document]
    retrieved_chunks: list[dict[str, Any]]
    used_web_search: bool
    answer: str
    sources: list[dict[str, str]]
    next_step: str


def _get_llm(max_tokens: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        api_key=config.GOOGLE_API_KEY,
        temperature=1.0,  # Prevents reasoning loops
        max_tokens=max_tokens,
        timeout=45.0,     # Prevents indefinite backend hangs
        convert_system_message_to_human=True,
        max_retries=1
    )


def _response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif hasattr(item, "get") and item.get("text"):
                parts.append(item.get("text"))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip()


def rewrite_query(state: GraphState) -> dict:
    try:
        question = state["question"]
        current_query = state.get("current_query", question)
        retry_count = state.get("retry_count", 0)

        prompt = (
            "Rewrite this technical question into one short search query for document retrieval. "
            "Keep important library or API terms. Return only the rewritten query.\n\n"
            f"Question: {question}"
        ) if retry_count == 0 else (
            "The first retrieval did not find relevant chunks. Write one broader alternative search query "
            "for technical documentation retrieval. Return only the new query.\n\n"
            f"Original question: {question}\n"
            f"Previous query: {current_query}"
        )

        rewritten_query = _response_to_text(_get_llm(80).invoke(prompt))
        return {"current_query": rewritten_query or current_query}
    except Exception as e:
        print(f"--- ERROR in rewrite_query: {e} ---")
        return {"current_query": state.get("current_query", state["question"])}


def retrieve_documents(state: GraphState) -> dict:
    try:
        vector_store = load_vector_store()
        query = state["current_query"]
        top_k = int(state.get("top_k", config.DEFAULT_TOP_K))
        
        results = vector_store.similarity_search_with_score(query, k=top_k)

        retrieved_docs: list[Document] = []
        retrieved_chunks: list[dict[str, Any]] = []

        for document, score in results:
            retrieved_docs.append(document)
            retrieved_chunks.append(
                {
                    "source": document.metadata.get("source", "Unknown"),
                    "content": document.page_content,
                    "score": float(score),
                    "grade": None,
                    "chunk_id": document.metadata.get("chunk_id"),
                }
            )

        return {
            "retrieved_docs": retrieved_docs,
            "retrieved_chunks": retrieved_chunks,
        }
    except Exception as e:
        print(f"--- ERROR in retrieve_documents: {e} ---")
        return {"retrieved_docs": [], "retrieved_chunks": []}


def _is_relevant(question: str, chunk_text: str) -> bool:
    prompt = (
        "You are grading a retrieved document chunk.\n"
        "Answer with only one word: relevant or irrelevant.\n"
        "Mark it relevant if it directly helps answer the technical question.\n\n"
        f"Question: {question}\n\n"
        f"Chunk:\n{chunk_text[:1200]}"
    )

    try:
        result = _response_to_text(_get_llm(10).invoke(prompt)).lower()
        return result.startswith("relevant")
    except Exception:
        return False


def grade_documents(state: GraphState) -> dict:
    try:
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        relevant_docs: list[Document] = []
        graded_chunks: list[dict[str, Any]] = []

        for document, chunk in zip(state.get("retrieved_docs", []), state.get("retrieved_chunks", []), strict=False):
            is_relevant = _is_relevant(question, document.page_content)
            chunk["grade"] = "relevant" if is_relevant else "irrelevant"
            graded_chunks.append(chunk)
            if is_relevant:
                relevant_docs.append(document)

        if relevant_docs:
            next_step = "generate_answer"
        elif retry_count < config.MAX_RETRIEVAL_RETRY:
            next_step = "retry"
            retry_count += 1
        else:
            next_step = "web_search"

        return {
            "relevant_docs": relevant_docs,
            "retrieved_chunks": graded_chunks,
            "retry_count": retry_count,
            "next_step": next_step,
        }
    except Exception as e:
        print(f"--- ERROR in grade_documents: {e} ---")
        return {"next_step": "web_search", "retry_count": state.get("retry_count", 0)}


def web_search(state: GraphState) -> dict:
    try:
        if not config.has_tavily_api_key():
            return {"web_results": [], "used_web_search": False}

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(
            query=state.get("current_query") or state["question"],
            search_depth="basic",
            topic="general",
            max_results=config.MAX_WEB_RESULTS,
            include_answer=False,
            include_raw_content=False,
        )

        web_results: list[Document] = []
        for item in response.get("results", []):
            content = (item.get("content") or "").strip()
            if not content:
                continue

            web_results.append(
                Document(
                    page_content=content[:1500],
                    metadata={
                        "source": item.get("url", "Web result"),
                        "title": item.get("title", "Web result"),
                        "type": "web",
                    },
                )
            )

        return {"web_results": web_results, "used_web_search": True}
    except Exception as e:
        print(f"--- ERROR in web_search: {e} ---")
        return {"web_results": [], "used_web_search": False}


def generate_answer(state: GraphState) -> dict:
    try:
        context_documents = state.get("relevant_docs") or state.get("web_results", [])
        if not context_documents:
            return {
                "answer": (
                    "I could not find enough relevant information in the indexed documents, "
                    "and no web fallback results were available."
                ),
                "sources": [],
            }

        context_blocks = []
        sources = []

        for index, document in enumerate(context_documents, start=1):
            source = document.metadata.get("source", "Unknown source")
            title = document.metadata.get("title") or source
            source_type = document.metadata.get("type", "document")
            sources.append({"title": title, "source": source, "type": source_type})
            context_blocks.append(f"[{index}] {title}\nSource: {source}\n{document.page_content}")

        prompt = (
            "Answer the technical question using only the context below.\n"
            "If the context is not enough, say so clearly.\n"
            "Keep the answer concise and practical.\n"
            "Use inline citations like [1] or [2] when making factual claims.\n\n"
            f"Question: {state['question']}\n\n"
            "Context:\n"
            f"{'\n\n'.join(context_blocks)}"
        )

        answer = _response_to_text(_get_llm(config.MAX_ANSWER_TOKENS).invoke(prompt))
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"--- ERROR in generate_answer: {e} ---")
        return {"answer": "Error generating answer.", "sources": []}


def route_after_grading(state: GraphState) -> str:
    return state.get("next_step", "web_search")


# IMPORTANT: This builds the graph and initializes the GRAPH variable
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_answer": "generate_answer",
            "retry": "rewrite_query",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

# This is the variable that was missing!
GRAPH = build_graph()


def run_query(question: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
    error_response = {
        "question": question,
        "rewritten_query": question,
        "answer": "An internal error occurred during processing.",
        "sources": [],
        "retrieved_chunks": [],
        "used_web_search": False,
        "retry_count": 0
    }

    try:
        final_state = GRAPH.invoke(
            {
                "question": question,
                "current_query": question,
                "top_k": top_k,
                "retry_count": 0,
                "used_web_search": False,
            }
        )

        return {
            "question": question,
            "rewritten_query": final_state.get("current_query", question),
            "answer": final_state.get("answer", "No answer generated."),
            "sources": final_state.get("sources", []),
            "retrieved_chunks": final_state.get("retrieved_chunks", []),
            "used_web_search": final_state.get("used_web_search", False),
            "retry_count": final_state.get("retry_count", 0),
        }

    except Exception as e:
        print(f"--- CRITICAL SYSTEM ERROR: {e} ---")
        error_response["answer"] = f"System error: {str(e)}"
        return error_response