"""
ingest.py
---------
Document ingestion pipeline: loads a PDF or crawls a URL, splits into chunks,
generates OpenAI embeddings, and persists the FAISS vectorstore to disk.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

from crawler import extract_documents_from_url

load_dotenv()

FAISS_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_db")


def _split_and_store(documents: list[Document]) -> None:
    """Shared helper: chunk → embed → merge/create FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_DB_PATH) and os.listdir(FAISS_DB_PATH):
        vectorstore = FAISS.load_local(
            FAISS_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_DB_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_DB_PATH)


def ingest_pdf(file_path: str) -> None:
    """
    Load a PDF, chunk it, embed it with OpenAI, and persist/merge into FAISS.

    Args:
        file_path: Absolute path to the PDF file on disk.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    _split_and_store(documents)


def ingest_url(url: str, max_child_urls: int = 30) -> int:
    """
    Crawl a URL (plus up to max_child_urls same-domain child pages),
    embed the content, and persist/merge into FAISS.

    Returns:
        Number of pages successfully crawled and indexed.
    """
    documents = extract_documents_from_url(url, max_child_urls=max_child_urls)
    if not documents:
        raise ValueError(f"No content could be extracted from: {url}")
    _split_and_store(documents)
    return len(documents)
