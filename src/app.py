# app.py

import streamlit as st
import os
import tempfile
# Imports from sibling files
from loader import process_file_to_document
from retriever import build_vector_store

# --- FINAL, STABLE LCEL IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 
# LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI


# --- 1. RAG Chain Setup ---

# Initialize LLM (relies on GEMINI_API_KEY environment variable)
# NOTE: This function is cached because the LLM object is stable.
@st.cache_resource(show_spinner=False)
def get_llm():
    """Initializes and caches the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Define the Prompt Template (using the stable ChatPromptTemplate)
QA_SYSTEM_PROMPT = (
    "You are a helpful academic assistant. Use the following retrieved context, "
    "including source and category information, to answer the user's question "
    "in a comprehensive, clear, and structured manner. "
    "If the answer is not in the context, clearly state that you cannot find "
    "the answer in the provided documents.\n\n"
    "Context:\n{context}"
)

# app.py (Corrected get_retrieval_rag_chain function)

# Note: This function is NOT cached.
def get_retrieval_rag_chain(llm, vector_store):
    """
    Builds the complete RAG chain using the fundamental LCEL pipe (|) operator.
    """
    
    # 1. Define the retriever component
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 2. Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    
    # 3. Define the chain that performs retrieval, combines context, and generates the answer
    rag_chain = (
        # --- FIX IS HERE ---
        # We assign the context key by first extracting the input string (x["input"])
        # and then piping that string into the retriever.
        RunnablePassthrough.assign(context=(lambda x: retriever.invoke(x["input"])))
        # The prompt step receives the full dictionary: {'input': 'question', 'context': [docs]}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return rag_chain


# --- 2. Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Smart Study Assistant", layout="wide")
    st.title("📚 Smart Study Assistant")
    st.caption("Upload your notes (PDF, DOCX, PPTX, TXT) and ask questions about the combined content.")

    # --- Authentication Check and Setup ---
    # Check for API Key in environment
    if "GEMINI_API_KEY" not in os.environ:
        if "GOOGLE_API_KEY" in os.environ:
            os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
        else:
            st.error("🚨 **GEMINI_API_KEY** environment variable not set. Please set your key in the secrets file.")
            return

    # CRITICAL: Force load the key right after the check to prevent the DefaultCredentialsError
    os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

    # Initialize session state for data storage
    if "study_docs" not in st.session_state:
        st.session_state.study_docs = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        
    # The LLM is retrieved from cache here
    llm = get_llm()


    # --- Sidebar for File Upload ---
    with st.sidebar:
        st.header("Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, PPTX, or TXT documents",
            type=["pdf", "docx", "pptx", "txt"],
            accept_multiple_files=True
        )
        
        # Logic to process new files and rebuild the vector store
        if uploaded_files and len(uploaded_files) != len(st.session_state.study_docs):
            
            new_docs = []
            with st.spinner("Processing documents and creating embeddings... (Free local model)"):
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily to disk for loader.py to access
                    file_extension = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        filepath = tmp_file.name
                        
                        # Use the loader function
                        doc = process_file_to_document(filepath)
                        if doc:
                            new_docs.append(doc)
                        
                        os.unlink(filepath) # Clean up temp file immediately
                        
            st.session_state.study_docs = new_docs
            
            # Build Vector Store from the processed documents (This function is cached in retriever.py)
            if st.session_state.study_docs:
                st.session_state.vector_store = build_vector_store(st.session_state.study_docs)
                st.success(f"✅ Successfully processed {len(st.session_state.study_docs)} documents!")
            else:
                st.warning("No valid text was extracted from the uploaded files.")
                st.session_state.vector_store = None
                    

        # Display currently loaded files
        st.header("Loaded Documents")
        if st.session_state.study_docs:
            for doc in st.session_state.study_docs:
                st.markdown(f"- **{doc.metadata['source']}** ({doc.metadata['category']})")
        else:
            st.info("Upload documents to start.")
            
        st.markdown("---")


    # --- Main Content Area for Querying ---
    
    if st.session_state.vector_store is not None:
        user_question = st.text_area("Ask a question about your study materials:", height=75)
        
        if user_question:
            
            # 1. Build a new RAG chain (not cached)
            # We pass the cached LLM and the cached vector store
            rag_chain = get_retrieval_rag_chain(llm, st.session_state.vector_store)
            
            # 2. Invoke the chain
            with st.spinner("Generating comprehensive answer..."):
                try:
                    # Invoke the LCEL chain with the user's question. 
                    response = rag_chain.invoke({"input": user_question}) 
                    
                    # Manually retrieve the chunks for source display (for simplicity)
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                    retrieved_docs = retriever.invoke(user_question)

                except Exception as e:
                    st.error(f"Error during generation: {e}")
                    st.exception(e)
                    return

            st.subheader("Answer")
            st.write(response)

            # Format retrieved documents for display
            context_str = "\n---\n".join([
                f"Source: {d.metadata.get('source', 'Unknown')}, Category: {d.metadata.get('category', 'N/A')}\nContent: {d.page_content}"
                for d in retrieved_docs
            ])

            with st.expander("Show Retrieved Source Chunks"):
                st.write(context_str)

    else:
        st.info("Please upload your study materials in the sidebar to begin asking questions.")


if __name__ == "__main__":
    main()
