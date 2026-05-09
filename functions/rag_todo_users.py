# rag_todo_users.py
# RAG-augmented todo extraction: use optional context (e.g. existing todos or retrieved docs)
# to improve extraction and avoid duplicates.

import json
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps and slow cold start
_todo_generator = None


def get_todo_generator():
    """Lazy load todo_generator module."""
    global _todo_generator
    if _todo_generator is None:
        import todo_generator as tg
        _todo_generator = tg
    return _todo_generator


def _format_context(context: Union[List[str], List[Dict[str, Any]]]) -> str:
    """Format context for inclusion in the user message."""
    if not context:
        return ""
    parts = []
    for i, item in enumerate(context[:50], 1):  # cap at 50 items
        if isinstance(item, str):
            parts.append(f"  {i}. {item}")
        elif isinstance(item, dict):
            title = item.get("title") or item.get("name") or ""
            detail = item.get("detail") or item.get("description") or ""
            date = item.get("date") or ""
            start = item.get("start") or ""
            line = title
            if detail:
                line += f" — {detail}"
            if date or start:
                line += f" ({date} {start})".strip()
            parts.append(f"  {i}. {line}" if line else f"  {i}. (existing todo)")
        else:
            parts.append(f"  {i}. {str(item)[:200]}")
    return "\n".join(parts)


def _get_context_for_user(user_id: Optional[str], query: str, top_k: int = 5) -> List[str]:
    """If user_id is set, retrieve context from FAISS user memory; else return empty list."""
    if not user_id:
        return []
    try:
        from user_memory import retrieve_user_context
        return retrieve_user_context(user_id=user_id, query=query, top_k=top_k)
    except Exception as e:
        logger.warning("Failed to retrieve user context from memory: %s", e)
        return []


def extract_todos_with_rag(
    user_input: str,
    context: Optional[Union[List[str], List[Dict[str, Any]]]] = None,
    user_id: Optional[str] = None,
    memory_top_k: int = 5,
    language: str = "thai",
    current_date: Optional[str] = None,
    timezone: str = "Asia/Bangkok",
) -> List[Dict[str, Any]]:
    """
    Extract structured todos from user input, optionally augmented with context (RAG).

    Context can be:
    - Passed explicitly as context (list of strings or todo-like dicts), or
    - Retrieved from FAISS user memory when user_id is provided.

    Args:
        user_input: Natural language description of one or more todos.
        context: Optional list of strings or todo-like dicts (e.g. existing todos).
        user_id: Optional; if set and context is not, retrieves from user_memory (FAISS).
        memory_top_k: Number of memory chunks to retrieve when using user_id (default 5).
        language: Language for processing.
        current_date: Current date in ISO format.
        timezone: User's timezone.

    Returns:
        List of structured todo dicts.
    """
    tg = get_todo_generator()
    # Resolve context: explicit first, else from FAISS when user_id present
    if context is None and user_id:
        context = _get_context_for_user(user_id, user_input, top_k=memory_top_k)
    if not context:
        return tg.extract_todo_from_text(
            user_input=user_input,
            language=language,
            current_date=current_date,
            timezone=timezone,
        )
    context_block = _format_context(context)
    augmented_input = (
        "=== Context (existing todos or relevant info; use to avoid duplicates and stay consistent) ===\n"
        f"{context_block}\n\n"
        "=== User input to extract new todos from ===\n"
        f"{user_input}"
    )
    logger.info("RAG todo extraction with context (%s items)", len(context))
    return tg.extract_todo_from_text(
        user_input=augmented_input,
        language=language,
        current_date=current_date,
        timezone=timezone,
    )
