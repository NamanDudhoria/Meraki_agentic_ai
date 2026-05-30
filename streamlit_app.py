import os
import uuid
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent import (
    build_vectorstore,
    create_rag_agent,
    forget_memory_record,
    remember_chat_interaction,
)

load_dotenv()


st.set_page_config(page_title="Meraki AI Consultant", page_icon=":soccer:", layout="wide")
st.title("Meraki AI Consultant")
st.caption("Agentic RAG over your Meraki history + live web search")


def _to_lc_messages(history: List[Dict[str, str]]):
    messages = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return build_vectorstore(rebuild=False)


@st.cache_resource(show_spinner=False)
def get_agent(session_id: str, memory_scope: str):
    include_all_memory = memory_scope == "All sessions"
    return create_rag_agent(
        get_vectorstore(),
        session_id=session_id,
        include_all_memory=include_all_memory,
    )


def reset_agent_cache():
    get_agent.clear()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "memory_record_ids" not in st.session_state:
    st.session_state.memory_record_ids = []


with st.sidebar:
    st.header("Settings")
    default_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key_input = st.text_input("Anthropic API Key", type="password", value=default_key, placeholder="sk-ant-...")
    model_name = st.text_input("Model", value=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    top_k = st.number_input("Retriever top-k", min_value=1, max_value=10, value=int(os.environ.get("MERAKI_TOP_K", "4")))
    learn_from_chat = st.toggle("Learn from this chat", value=True)
    memory_scope = st.radio("Memory scope", options=["All sessions", "This session only"], index=1)

    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input.strip()
    os.environ["ANTHROPIC_MODEL"] = str(model_name).strip()
    os.environ["MERAKI_TOP_K"] = str(int(top_k))

    col1, col2 = st.columns(2)
    if col1.button("Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding local vector index..."):
            build_vectorstore(rebuild=True)
            reset_agent_cache()
            get_vectorstore.clear()
            st.session_state.memory_record_ids = []
        st.success("Index rebuilt.")

    if col2.button("Forget Last Turn", use_container_width=True):
        # Remove last assistant and user messages from UI history.
        if st.session_state.chat_history:
            if st.session_state.chat_history[-1]["role"] == "assistant":
                st.session_state.chat_history.pop()
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                st.session_state.chat_history.pop()

        # Remove the latest learned memory record, if any.
        if st.session_state.memory_record_ids:
            last_memory_id = st.session_state.memory_record_ids.pop()
            forget_memory_record(get_vectorstore(), last_memory_id)
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Data sources: `Brand Solution Decks` and `Reports & Data`.")
    st.caption("OCR on scanned PDFs requires Poppler (`pdftoppm`) + Tesseract installed.")


if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning("Enter your Anthropic API key in the sidebar to start chatting.")
    st.stop()


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_prompt = st.chat_input("Ask for partnership strategy, competitor analysis, or funnel design...")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent = get_agent(st.session_state.session_id, memory_scope)
            lc_messages = _to_lc_messages(st.session_state.chat_history)
            result = agent.invoke({"messages": lc_messages})
            final = result["messages"][-1]
            answer = getattr(final, "content", str(final)).strip()
            st.markdown(answer)
            if learn_from_chat:
                memory_id = remember_chat_interaction(
                    get_vectorstore(),
                    user_message=user_prompt,
                    assistant_message=answer,
                    session_id=st.session_state.session_id,
                )
                if memory_id:
                    st.session_state.memory_record_ids.append(memory_id)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
