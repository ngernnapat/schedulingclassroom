"""Firebase Admin for the EVO mobile app (evoforluanching).

Cloud Functions for schedulingclassroom run in a different GCP project than
the React Native app. Coach / practice-step endpoints must verify ID tokens
and read/write user data in evoforluanching.

Set EVO_FIREBASE_SERVICE_ACCOUNT_JSON to a service-account key JSON string
for evoforluanching (Firebase console → Project settings → Service accounts).
Optional: EVO_FIREBASE_PROJECT_ID (default evoforluanching).
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Optional

import firebase_admin
from firebase_admin import auth as fb_auth
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

_EVO_APP_NAME = "evoforluanching"
_evo_app: Optional[Any] = None


def _load_sa_dict() -> Optional[dict]:
    """Service-account credentials for evoforluanching.

    Preference order:
      1. EVO_FIREBASE_SERVICE_ACCOUNT_JSON (Secret Manager / env) — production.
      2. EVO_FIREBASE_SERVICE_ACCOUNT_FILE (explicit path).
      3. Any bundled evoforluanching-firebase-adminsdk-*.json deployed beside
         this module — guarantees plan-context loads even when the secret
         was never set, which is what silently starved generate_task_content
         of planner intent.
    """
    raw = os.getenv("EVO_FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            return json.loads(raw)
        except Exception as e:
            logger.error("EVO_FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: %s", e)

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    explicit = os.getenv("EVO_FIREBASE_SERVICE_ACCOUNT_FILE", "").strip()
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        sorted(glob.glob(os.path.join(here, "evoforluanching-firebase-adminsdk-*.json")))
    )
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.info("EVO Firebase credentials loaded from file %s", os.path.basename(path))
            return data
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.error("EVO Firebase credentials file %s unreadable: %s", path, e)
    return None


def get_evo_app():
    global _evo_app
    if _evo_app is not None:
        return _evo_app
    project_id = os.getenv("EVO_FIREBASE_PROJECT_ID", "evoforluanching").strip()
    sa = _load_sa_dict()
    try:
        if sa:
            cred = credentials.Certificate(sa)
            _evo_app = firebase_admin.initialize_app(
                cred,
                name=_EVO_APP_NAME,
                options={"projectId": project_id},
            )
        else:
            logger.warning(
                "No EVO service-account credentials found (set "
                "EVO_FIREBASE_SERVICE_ACCOUNT_JSON or bundle the adminsdk JSON) "
                "— EVO token verify and Firestore are unavailable, so "
                "generate_task_content cannot load planner intent"
            )
            return None
        logger.info("EVO Firebase Admin ready (project=%s)", project_id)
    except ValueError:
        _evo_app = firebase_admin.get_app(_EVO_APP_NAME)
    except Exception as e:
        logger.error("EVO Firebase Admin init failed: %s", e)
        _evo_app = None
    return _evo_app


def evo_firestore():
    app = get_evo_app()
    return firestore.client(app=app) if app else None


def verify_evo_id_token(id_token: str) -> Optional[str]:
    app = get_evo_app()
    if not app or not id_token:
        return None
    try:
        decoded = fb_auth.verify_id_token(id_token, app=app)
        return decoded.get("uid")
    except Exception as e:
        logger.info("EVO id_token verify failed: %s", type(e).__name__)
        return None
