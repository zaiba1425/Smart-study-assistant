import streamlit as st # ADDED: Required for @st.cache_resource decorator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter # Previously fixed import
from langchain_core.documents import Document # Previously fixed import
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global embeddings instance (only load once)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Initializes and caches the local HuggingFace embeddings model."""
    # Note: 'sentence-transformers' must be installed for this to work
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_vector_store(documents: list[Document]) -> FAISS:
    """
    documents: list of LangChain Document objects
    returns: FAISS vector store
    """
    # Create embeddings object
    embeddings = get_embeddings()

    # Split documents into chunks for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split the list of LangChain documents
    docs = text_splitter.split_documents(documents)

    # Build FAISS vector store from the chunks
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def query_vector_store(vector_store: FAISS, query: str, k: int = 4) -> list[Document]:
    """
    vector_store: FAISS vector store
    query: string
    k: number of results (chunks) to retrieve
    returns: list of retrieved Document chunks
    """
    # The FAISS store already has the embeddings linked, no need to pass them again
    results = vector_store.similarity_search(query, k=k)
    return results