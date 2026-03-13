"""
chat_service.py
---------------
RAG retrieval chain: loads FAISS vectorstore, retrieves top-5 relevant chunks,
and generates an answer with source citations using LangChain runnables.
"""

from __future__ import annotations
import os
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from rag_pipeline import get_vectorstore

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an assistant for question answering tasks.
Use the retrieved context to answer the question.
If the answer is not contained in the context, say you don't know. Do not provide citations or sources when you don't know the answer.

Current date: {current_date}

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question: str) -> dict:
    """
    Run the RAG pipeline and return an answer with source citations.

    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "page": int, "excerpt": str}, ...]
        }
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return {
            "answer": "No documents have been uploaded yet. Please upload a PDF first.",
            "sources": [],
        }

    # Retrieve top-5 relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs: list[Document] = retriever.invoke(question)

    # Build the chain with current date
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    current_date = datetime.now().strftime("%B %d, %Y")

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
            "current_date": lambda _: current_date,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    # Check if the answer indicates uncertainty (don't provide sources)
    uncertainty_phrases = [
        "i don't know",
        "i do not know",
        "cannot determine",
        "not specified",
        "not mentioned",
        "not contained in the context",
        "unable to answer",
        "no information",
    ]
    
    answer_lower = answer.lower()
    should_include_sources = not any(phrase in answer_lower for phrase in uncertainty_phrases)

    # Build source citations only if answer is confident
    sources = []
    if should_include_sources:
        seen = set()
        for doc in retrieved_docs:
            meta = doc.metadata
            source_name = meta.get("source", "Unknown")
            page = meta.get("page", 0)
            key = (source_name, page)
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "source": os.path.basename(source_name) if source_name != "Unknown" else "Unknown",
                        "page": page + 1,  # convert 0-indexed to 1-indexed
                        "excerpt": doc.page_content[:300].strip(),
                    }
                )

    return {"answer": answer, "sources": sources}
