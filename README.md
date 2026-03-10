# DocuMind – RAG Knowledge Assistant

DocuMind is a **Retrieval-Augmented Generation (RAG) application** that allows users to upload documents or provide a website URL and ask questions about the content using natural language.

The system crawls websites (including internal links), processes documents, generates embeddings, and retrieves relevant context to produce accurate AI-powered answers.

---

## Features

- Upload **PDF, DOCX, TXT** files
- Ingest entire **websites with child pages**
- Automatic **text chunking**
- **Vector embeddings** for semantic search
- AI-powered **question answering**
- **Source references** from documents
- Simple **Streamlit UI**

---

## Architecture

```
User Query
   ↓
Embedding Generation
   ↓
Vector Search (ChromaDB)
   ↓
Relevant Context Retrieval
   ↓
LLM Response Generation
```

---

## Tech Stack

- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Vector Database:** ChromaDB  
- **Embeddings:** SentenceTransformers  
- **Web Crawling:** BeautifulSoup + Trafilatura  
- **LLM:** OpenAI / Local LLM  

---

## Project Structure

```
rag-documind
│
├── backend
│   ├── main.py
│   ├── crawler.py
│   ├── embeddings.py
│   └── rag_pipeline.py
│
├── frontend
│   └── app.py
│
├── vector_store
├── data
├── requirements.txt
└── README.md
```

---

## Setup

Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run backend

```
uvicorn backend.main:app --reload
```

Run frontend

```
streamlit run frontend/app.py
```

---

## Example

1. Enter a website URL  
2. System crawls and indexes all pages  
3. Ask a question

Example:

```
How does Python garbage collection work?
```

DocuMind retrieves relevant sections and generates an answer.

---

## Future Improvements

- Sitemap based crawling
- Multi-user document workspaces
- Chat history
- Hybrid search (vector + keyword)
- Scalable vector databases (Pinecone / Weaviate)

---

## License

MIT