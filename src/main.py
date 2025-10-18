import streamlit as st
import os
from loader import load_text_from_file
from categorizer import categorize_text
from retriever import build_vector_store, query_vector_store

st.set_page_config(page_title="Smart Study Assistant")

st.title("🧠 Smart Study Assistant")
st.write("Upload your notes, books, and presentations to get topic-based answers.")

uploaded_files = st.file_uploader("Upload your files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    for file in uploaded_files:
        filepath = os.path.join("docs", file.name)
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())

        text = load_text_from_file(filepath)
        if text:
            category = categorize_text(text)
            st.write(f"📂 {file.name} categorized as: **{category}**")
            all_texts.append(text)

    if st.button("Build Knowledge Base"):
        vectorstore = build_vector_store(all_texts)
        st.session_state['vectorstore'] = vectorstore
        st.success("✅ Knowledge base created!")

if "vectorstore" in st.session_state:
    query = st.text_input("Enter a topic or question:")
    if query:
        answer = query_vector_store(st.session_state['vectorstore'], query)
        st.write("### 🧾 Answer")
        st.write(answer)
