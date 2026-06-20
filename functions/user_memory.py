# user_memory.py
# FAISS-backed user memory: embed text, add/retrieve by user_id, and generate RAG responses.

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

# ---- CONFIG ----
EMBED_DIM = 1536  # text-embedding-3-small
EMBED_MODEL = "text-embedding-3-small"

# Lazy-loaded OpenAI client
_openai_client = None
_faiss_index = None
_metadata_store: List[Dict[str, Any]] = []


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        from openai_api_key import resolve_openai_api_key
        api_key = resolve_openai_api_key()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set (env/Secret Manager or Firestore ai_api_key/open-api-key)"
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_faiss_index():
    """Get or create the global FAISS index (lazy init to defer faiss import)."""
    global _faiss_index
    if _faiss_index is None:
        import faiss
        _faiss_index = faiss.IndexFlatL2(EMBED_DIM)
        logger.info("FAISS index initialized (dim=%s)", EMBED_DIM)
    return _faiss_index


def get_metadata_store() -> List[Dict[str, Any]]:
    """Return the global metadata store (list aligned with index vectors)."""
    return _metadata_store


# -----------------------------
# EMBEDDING
# -----------------------------
def embed_text(text: str) -> np.ndarray:
    """Embed a single text with OpenAI text-embedding-3-small."""
    if not text or not text.strip():
        raise ValueError("text must be non-empty")
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text.strip()
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -----------------------------
# ADD USER MEMORY
# -----------------------------
def add_memory(user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Add a memory for the user: embed text, add to FAISS, store metadata. Use metadata.get('todo_id') for later update/delete."""
    if not user_id or not text or not text.strip():
        raise ValueError("user_id and non-empty text are required")
    metadata = dict(metadata or {})
    metadata.setdefault("deleted", False)
    vector = embed_text(text)
    index = get_faiss_index()
    store = get_metadata_store()
    index.add(np.array([vector], dtype="float32"))
    store.append({
        "user_id": user_id,
        "text": text.strip(),
        **metadata
    })
    logger.info("Added memory for user_id=%s (total vectors=%s)", user_id, index.ntotal)


# -----------------------------
# EMBED TODO LISTS (bulk + new todos)
# -----------------------------
def _todo_to_memory_text(todo: Dict[str, Any]) -> str:
    """Format a single todo dict into a short text for embedding. Uses current title, detail, date, start, typeOfTodo — so when the todo is updated (e.g. title or date changed), re-embed with the full updated todo to refresh RAG."""
    title = (todo.get("title") or "").strip()
    #detail = (todo.get("detail") or "").strip()
    date = (todo.get("date") or "").strip()
    start = (todo.get("start") or "").strip()
    type_of = (todo.get("typeOfTodo") or "").strip()
    parts = [title]
    # if detail:
    #     parts.append(detail)
    if date:
        parts.append(date)
    if start:
        parts.append(start)
    if type_of:
        parts.append(type_of)
    text = " | ".join(parts)
    return text if text else str(todo)[:500]


def add_todos_as_memories(
    user_id: str,
    todos: List[Dict[str, Any]],
    mode: str = "per_todo",
) -> tuple:
    """
    Embed a list of todos into user memory (FAISS). Use for last-year bulk import and for new todos.

    Args:
        user_id: User identifier.
        todos: List of todo dicts (each with title, detail, date, start, typeOfTodo, etc.).
        mode: "per_todo" = one memory per todo (finer retrieval). "per_month" = one memory per month (group by date YYYY-MM).

    Returns:
        Tuple of (number of memories added, list of embedded text strings).
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("user_id is required")
    user_id = str(user_id).strip()
    if not todos or not isinstance(todos, list):
        return 0, []
    added = 0
    embedded_texts: List[str] = []
    if mode == "per_month":
        from collections import defaultdict
        by_month: Dict[str, List[str]] = defaultdict(list)
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            text = _todo_to_memory_text(todo)
            if not text:
                continue
            date_str = (todo.get("yearMonthStamp") or "").strip()[:7]  # YYYY-MM
            if not date_str or len(date_str) != 7:
                date_str = datetime.now().strftime("%Y-%m")
            by_month[date_str].append(text)
        for month, texts in sorted(by_month.items()):
            combined = f"{month} todos: " + "; ".join(texts[:50])  # cap items per month
            if combined.strip():
                add_memory(user_id, combined, metadata={"source": "todo_list", "month": month})
                embedded_texts.append(combined)
                added += 1
    else:
        for todo in todos:
            if not isinstance(todo, dict):
                continue
            text = _todo_to_memory_text(todo)
            if not text:
                continue
            date_str = (todo.get("date") or "").strip()[:10]
            todo_id = todo.get("id") or todo.get("todoId")  # for update/delete by id
            add_memory(user_id, text, metadata={
                "source": "todo",
                "date": date_str or None,
                "todo_id": todo_id,
            })
            embedded_texts.append(text)
            added += 1
    logger.info("Added %s todo memories for user_id=%s (mode=%s)", added, user_id, mode)
    return added, embedded_texts


# -----------------------------
# UPDATE / DELETE (soft delete so RAG stays in sync with todo changes)
# -----------------------------
def mark_memories_deleted_by_todo_ids(user_id: str, todo_ids: List[Any]) -> int:
    """
    Soft-delete memories that correspond to the given todo ids (e.g. when user updates or deletes todos).
    Only affects memories stored with metadata.todo_id (per_todo mode). Retrieval will skip these.

    Returns:
        Number of memories marked deleted.
    """
    if not user_id or not todo_ids:
        return 0
    user_id = str(user_id).strip()
    id_set = {str(tid) for tid in todo_ids if tid is not None}
    if not id_set:
        return 0
    store = get_metadata_store()
    count = 0
    for item in store:
        if item.get("user_id") != user_id:
            continue
        tid = item.get("todo_id")
        if tid is not None and str(tid) in id_set:
            item["deleted"] = True
            count += 1
    if count:
        logger.info("Marked %s memory/ies deleted for user_id=%s (todo_ids=%s)", count, user_id, list(id_set)[:5])
    return count


def retrieve_user_context(
    user_id: str,
    query: str,
    top_k: int = 5
) -> List[str]:
    """Retrieve top_k relevant memory texts for this user for the given query."""
    index = get_faiss_index()
    store = get_metadata_store()
    if index.ntotal == 0:
        return []
    query_vector = embed_text(query)
    # Search over all vectors, then filter by user_id to get top_k per-user
    k = min(index.ntotal, max(top_k * 10, 100))
    distances, indices = index.search(
        np.array([query_vector], dtype="float32"),
        k=k
    )
    results: List[str] = []
    for idx in indices[0]:
        if idx == -1:
            continue
        item = store[idx]
        if item.get("user_id") != user_id:
            continue
        if item.get("deleted"):
            continue
        results.append(item["text"])
        if len(results) >= top_k:
            break
    return results


# -----------------------------
# GENERATE LLM RESPONSE (lifestyle coach)
# -----------------------------
def generate_response(
    user_id: str,
    question: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> str:
    """Generate a response using retrieved user memory as context (RAG)."""
    context_chunks = retrieve_user_context(user_id, question)
    context_text = "\n".join(context_chunks) if context_chunks else "(No relevant memory yet.)"
    prompt = f"""You are an AI lifestyle coach.

User question:
{question}

Relevant memory:
{context_text}

Task:
1. Detect behavioral patterns
2. Give short, concrete suggestion
3. Avoid assumptions beyond context
"""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or "You are a thoughtful, evidence-based coach."},
            {"role": "user", "content": prompt}
        ]
    )
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("Empty response from model")
    return response.choices[0].message.content


# -----------------------------
# RETRIEVE MONTH CONTEXT (for RAG-driven month_context in planner)
# -----------------------------
def retrieve_month_context_from_rag(
    user_id: str,
    top_k_per_period: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve previous/current/next month context from user memory (RAG).
    Uses the user's stored todo-list memory to build month_context for text generation.

    Returns:
        Dict with keys "previous", "current", "next" (each str of joined chunks, or omitted if empty).
        None if user_id is empty or all retrievals are empty.
    """
    if not user_id or not str(user_id).strip():
        return None
    user_id = str(user_id).strip()
    month_context: Dict[str, Any] = {}
    current_month = datetime.now().strftime("%Y-%m")
    previous_month = (datetime.now() - timedelta(days=30)).strftime("%Y-%m")
    next_month = (datetime.now() + timedelta(days=30)).strftime("%Y-%m")
    queries = [
        ("previous", f"previous month todos {previous_month}"),
        ("current", f"current month todos {current_month}"),
        ("next", f"next month todos {next_month}"),
    ]
    for key, query in queries:
        chunks = retrieve_user_context(user_id, query, top_k=top_k_per_period)
        if chunks:
            month_context[key] = "\n".join(chunks)
    return month_context if month_context else None
