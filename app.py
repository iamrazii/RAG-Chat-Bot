
import streamlit as st
from BookChatBot import setup, build_chain, format_chat_history
import tempfile
import os

import nest_asyncio
nest_asyncio.apply()

st.set_page_config(page_title="📘 PDF ChatBot", layout="wide")
st.title(":blue_book: Upload PDF to Chat")

# Session states
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "history" not in st.session_state:
    st.session_state.history = []
if "final_chain" not in st.session_state:
    st.session_state.final_chain = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# ────── FILE UPLOAD ──────
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and not st.session_state.uploaded:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        chunks, retriever = setup(tmp_path)
        st.session_state.chunks = chunks
        st.session_state.final_chain = build_chain(chunks, retriever)
        st.session_state.uploaded = True

        os.remove(tmp_path)

    st.success("PDF processed successfully! You can start chatting now.")

# ────── CHAT UI ──────
if st.session_state.uploaded:
    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        chat_history = format_chat_history(st.session_state.history)
        with st.spinner("Thinking..."):
            response = st.session_state.final_chain.invoke({
                "question": user_input,
                "chat_history": chat_history
            })

        st.session_state.history.append((user_input, response))

    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
else:
    st.info("Please upload a PDF to start chatting.")