import streamlit as st
from langchain_google_genai import GoogleGenAIEmbeddings
from langchain_community.vectorstores import Chroma # CHANGED: Import ChromaDB for stability
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Model: Use Google's hosted embedding model (requires GEMINI_API_KEY)
EMBEDDING_MODEL_NAME = "text-embedding-004" 

# Global embeddings instance (only load once)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Initializes and caches the Google hosted embeddings model."""
    # This automatically uses the GEMINI_API_KEY saved in Streamlit secrets
    return GoogleGenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# CHANGE: Function signature now uses Chroma
def build_vector_store(documents: list[Document]) -> Chroma: 
    """
    documents: list of LangChain Document objects
    returns: ChromaDB vector store
    """
    embeddings = get_embeddings()

    # Split documents into chunks for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = text_splitter.split_documents(documents)

    # CHANGED: Build Chroma vector store instead of FAISS
    vector_store = Chroma.from_documents(docs, embeddings) 
    return vector_store

# CHANGE: Function signature now uses Chroma
def query_vector_store(vector_store: Chroma, query: str, k: int = 4) -> list[Document]:
    """
    Queries the vector store for relevant documents.
    """
    # Chroma uses the same similarity_search method
    results = vector_store.similarity_search(query, k=k)
    return results
