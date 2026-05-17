import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_request(method: str, path: str, payload: dict | None = None) -> dict | None:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, json=payload, timeout=120)
        if response.ok:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(detail)
        return None
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return None


def submit_feedback(rating: str) -> None:
    result = st.session_state.get("last_result")
    if not result:
        st.warning("Ask a question first.")
        return

    payload = {
        "question": result["question"],
        "answer": result["answer"],
        "rating": rating,
        "comment": st.session_state.get("feedback_comment", ""),
        "sources": [source["source"] for source in result.get("sources", [])],
    }
    response = api_request("POST", "/feedback", payload)
    if response:
        st.success("Feedback saved.")


st.set_page_config(page_title="RAG Docs Assistant", layout="wide")
st.title("RAG Technical Documentation Assistant")
st.caption("FastAPI + LangGraph + FAISS + Gemini")

with st.sidebar:
    st.subheader("Backend")
    st.write(f"API URL: `{API_BASE_URL}`")

    if st.button("Ingest documents", use_container_width=True):
        response = api_request("POST", "/ingest")
        if response:
            st.success(
                f"Indexed {response['document_count']} document(s) into {response['chunk_count']} chunk(s)."
            )

    documents_response = api_request("GET", "/documents")
    if documents_response:
        st.subheader("Documents")
        if documents_response["documents"]:
            for document in documents_response["documents"]:
                status = "indexed" if document["indexed"] else "not indexed yet"
                st.write(f"- {document['name']} ({status})")
        else:
            st.write("No documents found in `docs/` yet.")

question = st.text_input("Ask a question about your technical documents")
top_k = st.slider("Top-k chunks", min_value=1, max_value=8, value=4)

if st.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Running the RAG workflow..."):
            response = api_request(
                "POST",
                "/query",
                {"question": question.strip(), "top_k": top_k},
            )
        if response:
            st.session_state["last_result"] = response

result = st.session_state.get("last_result")
if result:
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    if result["sources"]:
        for index, source in enumerate(result["sources"], start=1):
            st.markdown(f"{index}. **{source['title']}**  \n`{source['source']}`")
    else:
        st.write("No sources returned.")

    st.subheader("Retrieved chunks")
    if result["retrieved_chunks"]:
        for chunk in result["retrieved_chunks"]:
            with st.expander(f"{chunk['source']} | {chunk.get('grade', 'ungraded')}"):
                if chunk.get("score") is not None:
                    st.caption(f"Vector score: {chunk['score']:.4f}")
                st.write(chunk["content"])
    else:
        st.write("No chunks retrieved.")

    st.subheader("Feedback")
    st.text_area("Optional comment", key="feedback_comment", height=100)
    left, right = st.columns(2)
    if left.button("Thumbs up", use_container_width=True):
        submit_feedback("up")
    if right.button("Thumbs down", use_container_width=True):
        submit_feedback("down")

    if result.get("used_web_search"):
        st.info("Web search fallback was used because the indexed chunks were not relevant enough.")
