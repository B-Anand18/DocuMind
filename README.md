# DocuMind – RAG Chatbot

A full-stack **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask natural language questions about them. Answers are grounded in your documents and include exact page citations.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.12) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector DB | FAISS (local, on disk) |
| Orchestration | LangChain |
| Frontend | Jinja2 + Vanilla HTML/CSS/JS |

---

## Project Structure

```
DocuMind/
│
├── backend/
│   ├── main.py            # FastAPI app – /upload and /chat routes
│   ├── ingest.py          # PDF loading, chunking, embedding & FAISS storage
│   ├── rag_pipeline.py    # FAISS loader utility
│   ├── chat_service.py    # RAG retrieval chain (LangChain runnables)
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html     # Jinja2 UI
│   ├── static/
│   │   └── styles.css
│   └── faiss_db/          # Persisted FAISS index (auto-created)
│
├── uploads/               # Uploaded PDFs (auto-created)
├── .env                   # OpenAI API key
└── README.md
```

---

## Setup

### 1. Add your OpenAI API key

Edit `.env` in the project root:

```
OPENAI_API_KEY=your_key_here
```

### 2. Create & activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 4. Run the server

```powershell
# from the backend/ directory
uvicorn main:app --reload
```

### 5. Open the app

Navigate to **http://localhost:8000** in your browser.

---

## Usage

1. Click **browse** (or drag & drop) in the sidebar to upload a PDF.  
2. Wait for the *"uploaded and indexed successfully"* confirmation.  
3. Type a question in the chat box and press **Enter** or the send button.  
4. Read the answer and expand the **Sources** cards to see exact page numbers and excerpts.

---

## Architecture

```
User Query
   ↓
OpenAI Embeddings (query)
   ↓
FAISS Vector Search  ←──── Persisted FAISS index
   ↓                              ↑
Top-5 Relevant Chunks        ingest.py  ←  PDF upload
   ↓
ChatPromptTemplate + ChatOpenAI (gpt-4o-mini)
   ↓
Answer + Source Citations
```

---

## License

MIT