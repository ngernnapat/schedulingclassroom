"""Resolve OpenAI API key from env or schedulingclassroom Firestore."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FIRESTORE_COLLECTION = "ai_api_key"
_FIRESTORE_DOCUMENT = "open-api-key"
_KEY_FIELDS = ("OPENAI_API_KEY", "openai_api_key", "api_key")
_cached_key: Optional[str] = None


def _extract_key_from_doc(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for field in _KEY_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _read_key_from_firestore() -> str:
    """Read from schedule-optimization Firestore (this project's default app)."""
    try:
        import firebase_admin
        from firebase_admin import firestore

        if not firebase_admin._apps:
            firebase_admin.initialize_app()

        doc = (
            firestore.client()
            .collection(_FIRESTORE_COLLECTION)
            .document(_FIRESTORE_DOCUMENT)
            .get()
        )
        if not doc.exists:
            logger.warning(
                "OpenAI key doc missing: %s/%s",
                _FIRESTORE_COLLECTION,
                _FIRESTORE_DOCUMENT,
            )
            return ""

        key = _extract_key_from_doc(doc.to_dict())
        if key:
            logger.info(
                "OpenAI API key loaded from Firestore %s/%s",
                _FIRESTORE_COLLECTION,
                _FIRESTORE_DOCUMENT,
            )
        return key
    except Exception as exc:
        logger.warning("Failed to load OpenAI API key from Firestore: %s", exc)
        return ""


def resolve_openai_api_key() -> str:
    """Env/Secret Manager first, then Firestore ai_api_key/open-api-key."""
    global _cached_key

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    if _cached_key:
        return _cached_key

    firestore_key = _read_key_from_firestore()
    if firestore_key:
        _cached_key = firestore_key
    return firestore_key
