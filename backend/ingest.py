"""
ingest.py
---------
Document ingestion pipeline: loads a PDF, splits it into chunks,
generates OpenAI embeddings, and persists the FAISS vectorstore to disk.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

FAISS_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_db")


def ingest_pdf(file_path: str) -> None:
    """
    Load a PDF, chunk it, embed it with OpenAI, and persist/merge into FAISS.

    Args:
        file_path: Absolute path to the PDF file on disk.
    """
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed & store / merge with existing FAISS index
    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_DB_PATH) and os.listdir(FAISS_DB_PATH):
        # Merge new chunks into the existing vectorstore
        vectorstore = FAISS.load_local(
            FAISS_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Persist to disk
    os.makedirs(FAISS_DB_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_DB_PATH)
