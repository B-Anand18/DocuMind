"""
main.py
-------
FastAPI application entry point.

Endpoints:
  GET  /            – Serves the Jinja2 HTML frontend
  POST /upload      – Accepts a PDF, saves it, and triggers ingestion
  POST /ingest-url  – Crawls a URL + child pages and indexes content
  POST /chat        – Accepts a question, runs RAG, returns answer + sources
"""

import os
import io

from typing import Union

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ingest import ingest_pdf, ingest_url, ingest_youtube
from backend.chat_service import answer_question
from backend.youtube_service import extract_youtube_data

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

app = FastAPI(title="DocuMind RAG Chatbot")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


class IngestUrlRequest(BaseModel):
    url: str
    max_child_urls: int = 30


class IngestYouTubeRequest(BaseModel):
    url: str


class SourceItem(BaseModel):
    source: str
    page: Union[int, str]  # Can be page number (int) or timestamp (str)
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat UI."""
    environment = os.getenv("ENVIRONMENT", "local")
    return templates.TemplateResponse(
        name="index.html", 
        context={"request": request, "environment": environment}, 
        request=request
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, process it in-memory, and trigger the ingestion pipeline.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    try:
        ingest_pdf(pdf_bytes, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {"message": f"'{file.filename}' uploaded and indexed successfully."}


@app.post("/ingest-url")
async def ingest_url_endpoint(body: IngestUrlRequest):
    """
    Crawl a URL and its same-domain child pages, then index the content into FAISS.
    """
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Enforce max_child_urls limit
    max_allowed = 30
    if body.max_child_urls > max_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"max_child_urls cannot exceed {max_allowed}. Requested: {body.max_child_urls}"
        )
    
    if body.max_child_urls < 1:
        raise HTTPException(status_code=400, detail="max_child_urls must be at least 1")

    try:
        pages_indexed = ingest_url(url, max_child_urls=body.max_child_urls)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {exc}") from exc

    return {"message": f"Crawled and indexed {pages_indexed} page(s) from '{url}'."}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """
    Accept a user question, run the RAG pipeline, and return the answer with sources.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = answer_question(body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    return result


@app.post("/ingest-youtube")
async def ingest_youtube_endpoint(body: IngestYouTubeRequest):
    """
    Extract transcript from a YouTube video and index it into FAISS.
    """
    url = body.url.strip()
    
    # Validate YouTube URL
    if not ("youtube.com" in url or "youtu.be" in url):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Must contain 'youtube.com' or 'youtu.be'"
        )
    
    try:
        # Extract video data
        video_data = extract_youtube_data(url)
        
        # Ingest into FAISS
        ingest_youtube(video_data)
        
        return {
            "message": f"YouTube video '{video_data['metadata']['title']}' indexed successfully.",
            "title": video_data["metadata"]["title"],
            "duration": video_data["metadata"]["duration"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YouTube ingestion failed: {exc}") from exc
