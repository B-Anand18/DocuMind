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
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from ingest import ingest_pdf, ingest_url
from chat_service import answer_question

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, "..", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

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


class SourceItem(BaseModel):
    source: str
    page: int
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
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, persist it to disk, and trigger the ingestion pipeline.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        ingest_pdf(dest_path)
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
