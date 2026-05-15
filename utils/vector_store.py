import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DB_PATH = "vectorstore/db_faiss"


def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def create_vector_store(chunks):

    embeddings = get_embeddings()

    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs("vectorstore", exist_ok=True)

    db.save_local(DB_PATH)

    return db


def load_vector_store():

    embeddings = get_embeddings()

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db