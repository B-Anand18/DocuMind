"""
ingest.py
---------
Document ingestion pipeline: loads a PDF or crawls a URL, splits into chunks,
generates OpenAI embeddings, and persists the FAISS vectorstore to disk.
"""

import os
import io
import tempfile
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from langsmith import traceable

from backend.crawler import extract_documents_from_url
from backend.youtube_service import extract_youtube_data

load_dotenv()

FAISS_DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_db")


def clear_vectorstore() -> None:
    """Delete the entire FAISS vectorstore to start fresh."""
    if os.path.exists(FAISS_DB_PATH):
        shutil.rmtree(FAISS_DB_PATH)
        print(f"Cleared vectorstore at {FAISS_DB_PATH}")


def _split_and_store(documents: list[Document]) -> None:
    """Shared helper: chunk → embed → create fresh FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_DB_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_DB_PATH)


@traceable(name="ingest_pdf")
def ingest_pdf(pdf_bytes: bytes, filename: str) -> None:
    """
    Clear previous data, then load a PDF from bytes, chunk it, embed it with OpenAI, and persist into FAISS.

    Args:
        pdf_bytes: PDF file content as bytes.
        filename: Original filename for metadata.
    """
    # Clear all previous documents
    clear_vectorstore()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Update metadata to use original filename
        for doc in documents:
            doc.metadata["source"] = filename
        
        _split_and_store(documents)
    finally:
        os.unlink(tmp_path)


@traceable(name="ingest_url")
def ingest_url(url: str, max_child_urls: int = 30) -> int:
    """
    Clear previous data, then crawl a URL (plus up to max_child_urls same-domain child pages),
    embed the content, and persist into FAISS.

    Returns:
        Number of pages successfully crawled and indexed.
    """
    # Clear all previous documents
    clear_vectorstore()
    
    documents = extract_documents_from_url(url, max_child_urls=max_child_urls)
    if not documents:
        raise ValueError(f"No content could be extracted from: {url}")
    _split_and_store(documents)
    return len(documents)


@traceable(name="ingest_youtube")
def ingest_youtube(video_data: dict) -> None:
    """
    Clear previous data, then ingest YouTube video transcript into FAISS.
    
    Args:
        video_data: Dictionary from extract_youtube_data() containing:
                   {"text": str, "metadata": dict, "segments": list, "video_id": str, "url": str}
    """
    # Clear all previous documents
    clear_vectorstore()
    
    # Create documents with timestamp metadata
    documents = []
    metadata = video_data["metadata"]
    
    # Group segments into larger chunks (every ~5 segments or ~30 seconds)
    segments = video_data["segments"]
    chunk_segments = []
    current_chunk = []
    current_duration = 0
    
    for seg in segments:
        current_chunk.append(seg)
        current_duration += len(seg["text"].split())
        
        # Create chunk every ~100 words or at end
        if current_duration >= 100 or seg == segments[-1]:
            if current_chunk:
                chunk_text = " ".join(s["text"] for s in current_chunk)
                start_time = current_chunk[0]["timestamp"]
                start_seconds = current_chunk[0]["start_seconds"]
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": metadata["title"],
                        "type": "youtube",
                        "timestamp": start_time,
                        "start_seconds": start_seconds,
                        "video_id": video_data["video_id"],
                        "url": video_data["url"],
                        "author": metadata["author"],
                    }
                )
                documents.append(doc)
                chunk_segments.append(current_chunk)
                current_chunk = []
                current_duration = 0
    
    if not documents:
        raise ValueError("No content could be extracted from YouTube video")
    
    _split_and_store(documents)
