"""Firebase Admin for the EVO mobile app (evoforluanching).

Cloud Functions for schedulingclassroom run in a different GCP project than
the React Native app. Coach / practice-step endpoints must verify ID tokens
and read/write user data in evoforluanching.

Set EVO_FIREBASE_SERVICE_ACCOUNT_JSON to a service-account key JSON string
for cross-project Firestore reads/writes. Token verification itself only
needs EVO's project ID and Google's public signing certificates, so signed-in
users can still authenticate safely when that optional secret is unavailable.
Optional: EVO_FIREBASE_PROJECT_ID (default evoforluanching).
"""

from __future__ import annotations

import ast
import glob
import json
import logging
import os
from typing import Any, Optional

import firebase_admin
import yaml
from firebase_admin import auth as fb_auth
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

_EVO_APP_NAME = "evoforluanching"
_evo_app: Optional[Any] = None
# Diagnostic record of how the EVO Admin app was authenticated. Both the
# plan-image Storage read and the coachSubscription Firestore read run
# cross-project against evoforluanching; when either returns 403 this tells us
# whether we authenticated with the EVO service account or silently fell back
# to schedule-optimization's Application Default credential (which has no
# access to evoforluanching and 403s on everything).
_evo_credential_kind: str = "uninitialized"
_evo_credential_principal: str = ""


def evo_credential_summary() -> str:
    """Human-readable description of the active EVO Admin credential."""
    principal = f" as {_evo_credential_principal}" if _evo_credential_principal else ""
    return f"{_evo_credential_kind}{principal}"


def _normalize_sa_dict(data: Any) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    normalized = dict(data)
    private_key = normalized.get("private_key")
    if isinstance(private_key, str):
        private_key = private_key.replace("\\n", "\n")
        begin = "-----BEGIN PRIVATE KEY-----"
        end = "-----END PRIVATE KEY-----"
        if begin in private_key and end in private_key:
            body = private_key.split(begin, 1)[1].split(end, 1)[0]
            compact = "".join(body.split())
            if compact:
                lines = [compact[index:index + 64] for index in range(0, len(compact), 64)]
                private_key = f"{begin}\n" + "\n".join(lines) + f"\n{end}\n"
        normalized["private_key"] = private_key
    return normalized


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
            return _normalize_sa_dict(json.loads(raw))
        except (TypeError, ValueError) as json_error:
            # Older secret versions were saved from ``str(dict)`` and use
            # single quotes. literal_eval accepts that exact data shape
            # without executing code; keep supporting it until the secret is
            # rotated to canonical JSON.
            try:
                legacy = ast.literal_eval(raw)
                if isinstance(legacy, dict):
                    logger.warning(
                        "EVO_FIREBASE_SERVICE_ACCOUNT_JSON uses legacy dict syntax; "
                        "rotate it to canonical JSON"
                    )
                    return _normalize_sa_dict(legacy)
            except (SyntaxError, ValueError):
                pass
            try:
                legacy = yaml.safe_load(raw)
                if isinstance(legacy, dict):
                    logger.warning(
                        "EVO_FIREBASE_SERVICE_ACCOUNT_JSON uses legacy YAML syntax; "
                        "rotate it to canonical JSON"
                    )
                    return _normalize_sa_dict(legacy)
            except yaml.YAMLError:
                pass
            logger.error(
                "EVO_FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: %s",
                json_error,
            )

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
                return _normalize_sa_dict(data)
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.error("EVO Firebase credentials file %s unreadable: %s", path, e)
    return None


def get_evo_app():
    global _evo_app, _evo_credential_kind, _evo_credential_principal
    if _evo_app is not None:
        return _evo_app
    project_id = os.getenv("EVO_FIREBASE_PROJECT_ID", "evoforluanching").strip()
    sa = _load_sa_dict()
    try:
        if sa:
            cred = credentials.Certificate(sa)
            _evo_credential_kind = "service_account"
            _evo_credential_principal = str(sa.get("client_email") or "")
            sa_project = str(sa.get("project_id") or "")
            if sa_project and sa_project != project_id:
                # A service account whose project differs from evoforluanching
                # will 403 on Storage/Firestore even though it is a valid key.
                logger.warning(
                    "EVO service account project_id=%s does not match target "
                    "project=%s; cross-project reads will be Forbidden",
                    sa_project,
                    project_id,
                )
            _evo_app = firebase_admin.initialize_app(
                cred,
                name=_EVO_APP_NAME,
                options={"projectId": project_id},
            )
        else:
            _evo_credential_kind = "application_default_fallback"
            logger.warning(
                "No EVO service-account credentials found (set "
                "EVO_FIREBASE_SERVICE_ACCOUNT_JSON or bundle the adminsdk JSON) "
                "— verifying EVO ID tokens with public signing certificates; "
                "cross-project Firestore access may be unavailable"
            )
            # Firebase ID-token verification checks the token's signature,
            # audience, issuer, expiry, and subject using Google's public
            # certificates. It does not require a private service-account key.
            # Cloud Run's application-default credential remains attached only
            # because initialize_app requires a credential object.
            _evo_app = firebase_admin.initialize_app(
                credentials.ApplicationDefault(),
                name=_EVO_APP_NAME,
                options={"projectId": project_id},
            )
        logger.info(
            "EVO Firebase Admin ready (project=%s, credential=%s)",
            project_id,
            evo_credential_summary(),
        )
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
