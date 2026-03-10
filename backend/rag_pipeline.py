"""
rag_pipeline.py
---------------
Utility to load the persisted FAISS vectorstore from disk.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

FAISS_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_db")


def get_vectorstore() -> FAISS | None:
    """Load FAISS vectorstore from disk. Returns None if it does not exist yet."""
    if not os.path.exists(FAISS_DB_PATH) or not os.listdir(FAISS_DB_PATH):
        return None

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
