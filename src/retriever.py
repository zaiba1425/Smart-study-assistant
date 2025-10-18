from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def build_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

def query_vector_store(vectorstore, query):
    docs = vectorstore.similarity_search(query, k=3)
    results = [d.page_content for d in docs]
    return "\n\n".join(results)
