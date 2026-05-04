"""
chat_service.py
---------------
RAG retrieval chain: loads FAISS vectorstore, retrieves relevant chunks dynamically,
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
from langsmith import traceable

from backend.rag_pipeline import get_vectorstore

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an assistant for question answering tasks.
Use the retrieved context to answer the question.

IMPORTANT INSTRUCTIONS:
- If the question asks for a LIST, SUMMARY, or OVERVIEW, compile ALL relevant information from the context
- For "list all" or "what are all" questions, extract and enumerate every item mentioned
- For summary questions, provide a comprehensive overview covering all main points
- If the answer is not contained in the context, say you don't know. Do not provide citations or sources when you don't know the answer.

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


def classify_query_type(question: str) -> tuple[str, int]:
    """
    Classify query type and return appropriate chunk count.
    
    Returns:
        (query_type, chunk_count)
        - 'comprehensive': 20 chunks (for lists, summaries, overviews)
        - 'comparison': 10 chunks (for comparisons, differences)
        - 'specific': 5 chunks (for specific questions)
    """
    question_lower = question.lower()
    
    # Comprehensive queries - need lots of context
    comprehensive_keywords = [
        'list all', 'list the', 'what are all', 'show all',
        'summarize', 'summary', 'overview', 'explain everything',
        'all the', 'every', 'complete list', 'enumerate',
        'what topics', 'what questions', 'cover', 'discuss',
        'what are the', 'give me all', 'tell me all',
        'main points', 'key points', 'all questions',
    ]
    
    # Comparison queries - need moderate context
    comparison_keywords = [
        'compare', 'difference between', 'vs', 'versus',
        'contrast', 'similar', 'different', 'comparison'
    ]
    
    # Check comprehensive
    if any(keyword in question_lower for keyword in comprehensive_keywords):
        return ('comprehensive', 20)
    
    # Check comparison
    if any(keyword in question_lower for keyword in comparison_keywords):
        return ('comparison', 10)
    
    # Default: specific query
    return ('specific', 5)


@traceable(name="answer_question")
def answer_question(question: str) -> dict:
    """
    Run the RAG pipeline and return an answer with source citations.
    Uses dynamic chunk retrieval based on query type.

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

    # Classify query and get appropriate chunk count
    query_type, chunk_count = classify_query_type(question)
    print(f"[Query Classification] Type: {query_type}, Retrieving {chunk_count} chunks")

    # Retrieve with dynamic k value
    retriever = vectorstore.as_retriever(search_kwargs={"k": chunk_count})
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
            source_type = meta.get("type", "pdf")
            
            # Skip citations for YouTube videos
            if source_type == "youtube":
                continue
            
            # Only show citations for PDF or URL sources
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
