# personalization_context.py
# Builds bounded personalization blocks for LLM prompts from behavior signals,
# today's schedule, RAG memories, and intent profile. See docs/ORCHESTRATION.md.

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Character budgets (~4 chars/token). Keeps prompts fast and within Cloud Function limits.
MAX_PROMPT_BLOCK_CHARS = 2500
MAX_TODAY_TODOS = 8
MAX_TITLE_CHARS = 40
MAX_DETAIL_CHARS = 100
MAX_RAG_CHUNKS = 5
MAX_RAG_CHUNK_CHARS = 200
MAX_INTENT_ITEMS = 3
MAX_LIFE_MONTH_SAMPLE_TITLES = 4
MAX_LIFE_MONTH_TOP_TYPES = 3


def _shift_year_month(year_month: str, delta_months: int) -> str:
    """Return YYYY-MM shifted by delta_months (can be negative)."""
    try:
        parts = str(year_month).strip().split("-")
        year = int(parts[0])
        month = int(parts[1])
    except (IndexError, ValueError, TypeError):
        now = datetime.now(ZoneInfo("Asia/Bangkok"))
        year, month = now.year, now.month
    month_idx = (year * 12 + (month - 1)) + delta_months
    new_year = month_idx // 12
    new_month = (month_idx % 12) + 1
    return f"{new_year:04d}-{new_month:02d}"


def _todo_completed(todo: Dict[str, Any]) -> bool:
    if todo.get("completed") is True or todo.get("isCompleted") is True:
        return True
    return False


def refine_month_todos_snapshot(
    todos: List[Dict[str, Any]],
    year_month: str,
    role: str,
) -> Dict[str, Any]:
    """
    Compress raw month todos into a small stats snapshot for LLM context.
    No LLM call — deterministic ingest/refine step.
    """
    active = [t for t in todos if isinstance(t, dict)]
    scheduled = len(active)
    completed = sum(1 for t in active if _todo_completed(t))
    completion_rate = round(completed / scheduled, 2) if scheduled else None

    by_date: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    plan_linked = 0
    for task in active:
        day = str(task.get("date") or "").strip()
        if day:
            by_date[day] = by_date.get(day, 0) + 1
        todo_type = str(task.get("typeOfTodo") or "general").strip() or "general"
        type_counts[todo_type] = type_counts.get(todo_type, 0) + 1
        if task.get("planId") or task.get("planName"):
            plan_linked += 1

    busiest_day = None
    if by_date:
        busiest_date, busiest_count = max(by_date.items(), key=lambda item: item[1])
        busiest_day = {"date": busiest_date, "count": busiest_count}

    top_types = [
        name
        for name, _ in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[
            :MAX_LIFE_MONTH_TOP_TYPES
        ]
    ]

    def _title(task: Dict[str, Any]) -> str:
        return _truncate(task.get("title") or "Task", MAX_TITLE_CHARS)

    pending = [t for t in active if not _todo_completed(t)]
    done = [t for t in active if _todo_completed(t)]

    if role == "next":
        pending.sort(key=lambda t: str(t.get("date") or ""))
        sample_pool = pending[:12] or active[:8]
    elif role == "previous":
        done.sort(key=lambda t: str(t.get("date") or ""), reverse=True)
        sample_pool = done[:10] or active[:8]
    else:
        pending.sort(key=lambda t: str(t.get("date") or ""))
        sample_pool = (pending[:6] + done[:4]) or active[:8]

    seen_titles: set[str] = set()
    sample_titles: List[str] = []
    for task in sample_pool:
        title = _title(task)
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        sample_titles.append(title)
        if len(sample_titles) >= MAX_LIFE_MONTH_SAMPLE_TITLES:
            break

    return {
        "year_month": year_month,
        "role": role,
        "scheduled": scheduled,
        "completed": completed,
        "completion_rate": completion_rate,
        "active_days": len(by_date),
        "busiest_day": busiest_day,
        "top_types": top_types,
        "sample_titles": sample_titles,
        "plan_linked": plan_linked,
    }


def _one_line_month_summary(snapshot: Dict[str, Any]) -> str:
    ym = snapshot.get("year_month") or "?"
    role = snapshot.get("role") or "current"
    labels = {"previous": "Previous", "current": "Current", "next": "Next"}
    prefix = labels.get(role, "Month")
    sched = int(snapshot.get("scheduled") or 0)
    if sched == 0:
        return f"{prefix} ({ym}): no scheduled tasks"

    detail_parts = [f"{sched} tasks"]
    rate = snapshot.get("completion_rate")
    if role == "next":
        detail_parts.append("upcoming")
    elif isinstance(rate, (int, float)):
        detail_parts.append(f"{int(float(rate) * 100)}% done")

    active_days = snapshot.get("active_days")
    if isinstance(active_days, int) and active_days > 0:
        detail_parts.append(f"{active_days} active days")

    busiest = snapshot.get("busiest_day")
    if isinstance(busiest, dict) and busiest.get("date"):
        detail_parts.append(f"busiest {busiest['date']} ({busiest.get('count', 0)})")

    top_types = snapshot.get("top_types")
    if isinstance(top_types, list) and top_types:
        detail_parts.append("themes: " + ", ".join(str(t) for t in top_types[:3]))

    plan_linked = snapshot.get("plan_linked")
    if isinstance(plan_linked, int) and plan_linked > 0:
        detail_parts.append(f"{plan_linked} plan-linked")

    samples = snapshot.get("sample_titles")
    if isinstance(samples, list) and samples:
        detail_parts.append("e.g. " + ", ".join(str(s) for s in samples[:3]))

    return f"{prefix} ({ym}): " + "; ".join(detail_parts)


def format_life_months_block(snapshots: List[Dict[str, Any]]) -> str:
    """Render refined prev/current/next month snapshots as a compact block."""
    if not snapshots:
        return ""
    lines = [_one_line_month_summary(snap) for snap in snapshots if isinstance(snap, dict)]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    return (
        "Life calendar (refined prev / current / next month — use for rhythm, load, and direction):\n"
        + "\n".join(f"• {ln}" for ln in lines)
    )


def format_month_life_from_client(summary: Any) -> str:
    """Accept client-precomputed month_life_summary {previous, current, next} strings or snapshots."""
    if isinstance(summary, str) and summary.strip():
        return _truncate(summary.strip(), 900)
    if not isinstance(summary, dict):
        return ""
    snapshots: List[Dict[str, Any]] = []
    for role in ("previous", "current", "next"):
        val = summary.get(role)
        if isinstance(val, str) and val.strip():
            snapshots.append({"role": role, "year_month": role, "scheduled": 1, "sample_titles": [val.strip()]})
        elif isinstance(val, dict):
            val = dict(val)
            val.setdefault("role", role)
            snapshots.append(val)
    return format_life_months_block(snapshots)


def fetch_month_todos_from_firestore(user_id: str, year_month: str) -> List[Dict[str, Any]]:
    if not user_id or not year_month:
        return []
    try:
        from firebase_admin import firestore

        uid = str(user_id).strip()
        snap = (
            firestore.client()
            .collection("users")
            .document(uid)
            .collection("to-do-posts")
            .where("yearMonthStamp", "==", year_month)
            .get()
        )
        rows: List[Dict[str, Any]] = []
        for doc in snap:
            row = doc.to_dict() or {}
            rows.append(
                {
                    "title": row.get("title"),
                    "date": row.get("date"),
                    "start": row.get("start"),
                    "completed": row.get("completed"),
                    "isCompleted": row.get("isCompleted"),
                    "typeOfTodo": row.get("typeOfTodo"),
                    "planId": row.get("planId"),
                    "planName": row.get("planName"),
                }
            )
        if rows:
            return rows
    except Exception as exc:
        logger.warning(
            "fetch_month_todos_from_firestore (stamp) failed for %s %s: %s",
            user_id,
            year_month,
            exc,
        )

    try:
        from firebase_admin import firestore

        parts = str(year_month).split("-")
        year = int(parts[0])
        month = int(parts[1])
        import calendar as cal_mod

        last_day = cal_mod.monthrange(year, month)[1]
        start = f"{year_month}-01"
        end = f"{year_month}-{last_day:02d}"
        snap = (
            firestore.client()
            .collection("users")
            .document(str(user_id).strip())
            .collection("to-do-posts")
            .where("date", ">=", start)
            .where("date", "<=", end)
            .get()
        )
        rows = []
        for doc in snap:
            row = doc.to_dict() or {}
            rows.append(
                {
                    "title": row.get("title"),
                    "date": row.get("date"),
                    "start": row.get("start"),
                    "completed": row.get("completed"),
                    "isCompleted": row.get("isCompleted"),
                    "typeOfTodo": row.get("typeOfTodo"),
                    "planId": row.get("planId"),
                    "planName": row.get("planName"),
                }
            )
        return rows
    except Exception as exc:
        logger.warning(
            "fetch_month_todos_from_firestore (date) failed for %s %s: %s",
            user_id,
            year_month,
            exc,
        )
        return []


def build_life_month_context_from_firestore(
    user_id: str,
    anchor_year_month: Optional[str] = None,
) -> str:
    """Fetch prev/current/next month todos, refine, return compact text block."""
    if not user_id or not str(user_id).strip():
        return ""
    anchor = str(anchor_year_month).strip() if anchor_year_month else ""
    if not anchor or len(anchor) < 7:
        anchor = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m")

    months = {
        "previous": _shift_year_month(anchor, -1),
        "current": anchor,
        "next": _shift_year_month(anchor, 1),
    }

    try:
        from firebase_admin import firestore

        uid = str(user_id).strip()
        stamp_list = list(months.values())
        snap = (
            firestore.client()
            .collection("users")
            .document(uid)
            .collection("to-do-posts")
            .where("yearMonthStamp", "in", stamp_list)
            .get()
        )
        by_month: Dict[str, List[Dict[str, Any]]] = {ym: [] for ym in stamp_list}
        for doc in snap:
            row = doc.to_dict() or {}
            ym = str(row.get("yearMonthStamp") or "").strip()
            if ym not in by_month:
                continue
            by_month[ym].append(
                {
                    "title": row.get("title"),
                    "date": row.get("date"),
                    "start": row.get("start"),
                    "completed": row.get("completed"),
                    "isCompleted": row.get("isCompleted"),
                    "typeOfTodo": row.get("typeOfTodo"),
                    "planId": row.get("planId"),
                    "planName": row.get("planName"),
                }
            )
    except Exception as exc:
        logger.warning("build_life_month_context batch fetch failed: %s", exc)
        by_month = {
            ym: fetch_month_todos_from_firestore(user_id, ym) for ym in months.values()
        }

    snapshots = [
        refine_month_todos_snapshot(by_month[months[role]], months[role], role)
        for role in ("previous", "current", "next")
    ]
    return format_life_months_block(snapshots)


def _truncate(text: Any, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _safe_todo_id(todo: Dict[str, Any]) -> Optional[str]:
    for key in ("todoID", "todoId", "id"):
        val = todo.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def format_todo_line(todo: Dict[str, Any], *, include_detail: bool = False) -> str:
    """Single todo as a compact line for LLM consumption."""
    if not isinstance(todo, dict):
        return ""
    title = _truncate(todo.get("title") or "Task", MAX_TITLE_CHARS)
    start = str(todo.get("start") or "").strip()
    completed = todo.get("completed")
    status = ""
    if isinstance(completed, bool):
        status = " ✓" if completed else ""
    line = f"• {start} {title}{status}".strip() if start else f"• {title}{status}".strip()
    if include_detail:
        detail = _truncate(todo.get("detail"), MAX_DETAIL_CHARS)
        if detail:
            line += f" — {detail}"
    type_of = str(todo.get("typeOfTodo") or "").strip()
    if type_of:
        line += f" [{type_of}]"
    return line


def format_today_todos_summary(
    todos: Optional[List[Dict[str, Any]]],
    *,
    current_todo_id: Optional[str] = None,
    max_items: int = MAX_TODAY_TODOS,
) -> str:
    """Format today's todos with caps. Highlights the active todo when id is known."""
    if not todos or not isinstance(todos, list):
        return ""

    active = [t for t in todos if isinstance(t, dict)]
    if not active:
        return ""

    def _sort_key(task: Dict[str, Any]) -> tuple:
        start = str(task.get("start") or "").strip()
        if not start:
            return (1, 9999)
        parts = start.split(":")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1][:2].isdigit():
            return (0, int(parts[0]) * 60 + int(parts[1][:2]))
        return (0, start)

    active.sort(key=_sort_key)
    lines: List[str] = []
    shown = 0
    for task in active:
        if shown >= max_items:
            break
        tid = _safe_todo_id(task)
        prefix = "→ " if current_todo_id and tid == current_todo_id else ""
        line = format_todo_line(task, include_detail=(tid == current_todo_id))
        if line:
            lines.append(prefix + line)
            shown += 1

    remaining = len(active) - shown
    if remaining > 0:
        lines.append(f"• +{remaining} more today")
    if not lines:
        return ""
    return f"Today's schedule ({len(active)}):\n" + "\n".join(lines)


def format_intent_profile_block(profile: Optional[Dict[str, Any]]) -> str:
    if not profile or not isinstance(profile, dict):
        return ""
    parts: List[str] = []
    top_cats = profile.get("topIntentCategories")
    if isinstance(top_cats, list) and top_cats:
        parts.append("Top focus areas: " + ", ".join(str(c) for c in top_cats[:MAX_INTENT_ITEMS]))
    time_blocks = profile.get("preferredTimeBlocks")
    if isinstance(time_blocks, list) and time_blocks:
        parts.append("Preferred times: " + ", ".join(str(t) for t in time_blocks[:MAX_INTENT_ITEMS]))
    rate = profile.get("completionPattern", {})
    if isinstance(rate, dict):
        cr = rate.get("completionRate")
        if isinstance(cr, (int, float)):
            parts.append(f"Recent completion rate: {int(round(cr * 100))}%")
    if not parts:
        return ""
    return "Intent signals:\n" + "\n".join(f"  • {p}" for p in parts)


def format_rag_block(chunks: Optional[List[str]]) -> str:
    if not chunks:
        return ""
    lines = []
    for chunk in chunks[:MAX_RAG_CHUNKS]:
        if isinstance(chunk, str) and chunk.strip():
            lines.append(f"  • {_truncate(chunk.strip(), MAX_RAG_CHUNK_CHARS)}")
    if not lines:
        return ""
    return "Relevant past activity:\n" + "\n".join(lines)


def format_active_plans_block(plans: List[Dict[str, Any]], *, max_items: int = 3) -> str:
    """Compact summary of lifestyle plans the user is following."""
    if not isinstance(plans, list) or not plans:
        return ""
    lines: List[str] = []
    for plan in plans[:max_items]:
        if not isinstance(plan, dict):
            continue
        name = _truncate(plan.get("planName") or plan.get("name") or "Plan", MAX_TITLE_CHARS)
        detail = _truncate(plan.get("detailPrompt") or plan.get("detail") or "", MAX_DETAIL_CHARS)
        category = _truncate(plan.get("category") or "", 24)
        bits = [name]
        if category:
            bits.append(f"({category})")
        line = " ".join(bits)
        if detail:
            line += f" — {detail}"
        lines.append(f"• {line}")
    if not lines:
        return ""
    remaining = len(plans) - min(len(plans), max_items)
    if remaining > 0:
        lines.append(f"• +{remaining} more active plan(s)")
    return "Active lifestyle plans:\n" + "\n".join(lines)


def fetch_active_plans_from_firestore(user_id: str, *, max_items: int = 3) -> List[Dict[str, Any]]:
    """Load non-draft lifestyle plans the user is following."""
    if not user_id or not str(user_id).strip():
        return []
    try:
        from firebase_admin import firestore

        uid = str(user_id).strip()
        snap = (
            firestore.client()
            .collection("users")
            .document(uid)
            .collection("lifestyle-plans")
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(max_items * 2)
            .get()
        )
        plans: List[Dict[str, Any]] = []
        for doc in snap:
            row = doc.to_dict() or {}
            if row.get("isDraft") is True:
                continue
            plans.append(
                {
                    "planId": row.get("planId") or doc.id,
                    "planName": row.get("planName"),
                    "detailPrompt": row.get("detailPrompt") or row.get("detail"),
                    "category": row.get("category"),
                }
            )
            if len(plans) >= max_items:
                break
        return plans
    except Exception as exc:
        logger.warning("fetch_active_plans_from_firestore failed for %s: %s", user_id, exc)
        return []


def assemble_personalization_block(
    *,
    identity_block: str = "",
    life_months_block: str = "",
    plans_block: str = "",
    today_block: str = "",
    rag_block: str = "",
    intent_block: str = "",
) -> str:
    """Merge sections and enforce total character budget."""
    blocks = {
        "identity": identity_block.strip() if identity_block else "",
        "life_months": life_months_block.strip() if life_months_block else "",
        "plans": plans_block.strip() if plans_block else "",
        "today": today_block.strip() if today_block else "",
        "rag": rag_block.strip() if rag_block else "",
        "intent": intent_block.strip() if intent_block else "",
    }
    # Drop lowest-priority sections until within budget.
    drop_order = ("intent", "rag", "plans", "today", "life_months")

    while True:
        sections = [b for b in blocks.values() if b]
        if not sections:
            return ""

        combined = (
            "=== PERSONALIZATION (ground advice in this — do not invent facts) ===\n"
            + "\n\n".join(sections)
            + "\n=== END PERSONALIZATION ==="
        )
        if len(combined) <= MAX_PROMPT_BLOCK_CHARS:
            return combined

        dropped = False
        for key in drop_order:
            if blocks[key]:
                blocks[key] = ""
                dropped = True
                break
        if not dropped:
            return combined[:MAX_PROMPT_BLOCK_CHARS] + "…"


def _today_ymd_in_timezone(tz_name: Optional[str]) -> str:
    try:
        tz = ZoneInfo(tz_name) if tz_name else ZoneInfo("Asia/Bangkok")
    except Exception:
        tz = ZoneInfo("Asia/Bangkok")
    return datetime.now(tz).strftime("%Y-%m-%d")


def fetch_identity_context_from_firestore(user_id: str) -> Optional[Dict[str, Any]]:
    """Read users/{uid}/identityProfile/current (matches backend fetchIdentityContext shape)."""
    if not user_id or not str(user_id).strip():
        return None
    try:
        from firebase_admin import firestore

        db = firestore.client()
        uid = str(user_id).strip()
        profile_snap = (
            db.collection("users")
            .document(uid)
            .collection("identityProfile")
            .document("current")
            .get()
        )
        profile = profile_snap.to_dict() if profile_snap.exists else {}

        latest_badge = None
        badges = (
            db.collection("users")
            .document(uid)
            .collection("identityBadges")
            .order_by("awardedAt", direction=firestore.Query.DESCENDING)
            .limit(1)
            .get()
        )
        if badges:
            b = badges[0].to_dict() or {}
            awarded = b.get("awardedAt")
            awarded_iso = awarded.isoformat() if hasattr(awarded, "isoformat") else None
            latest_badge = {
                "threshold": b.get("threshold"),
                "title": b.get("title"),
                "becomingPhrase": b.get("becomingPhrase"),
                "awardedAt": awarded_iso,
            }

        if not profile and not latest_badge:
            return None

        return {
            "currentStreak": profile.get("currentStreak") or 0,
            "longestStreak": profile.get("longestStreak") or 0,
            "lastCompletionDate": profile.get("lastCompletionDate"),
            "dayOfWeek": profile.get("dayOfWeek") if isinstance(profile.get("dayOfWeek"), list) else None,
            "latestBadge": latest_badge,
            "unlockedBadges": profile.get("unlockedBadges")
            if isinstance(profile.get("unlockedBadges"), list)
            else [],
        }
    except Exception as exc:
        logger.warning("fetch_identity_context_from_firestore failed for %s: %s", user_id, exc)
        return None


def fetch_intent_profile_from_firestore(user_id: str) -> Optional[Dict[str, Any]]:
    if not user_id or not str(user_id).strip():
        return None
    try:
        from firebase_admin import firestore

        snap = firestore.client().collection("users").document(str(user_id).strip()).get()
        if not snap.exists:
            return None
        profile = (snap.to_dict() or {}).get("intent_profile")
        return profile if isinstance(profile, dict) else None
    except Exception as exc:
        logger.warning("fetch_intent_profile_from_firestore failed for %s: %s", user_id, exc)
        return None


def fetch_today_todos_from_firestore(user_id: str, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load today's todos for personalization (title, time, completion only)."""
    if not user_id or not str(user_id).strip():
        return []
    try:
        from firebase_admin import firestore

        db = firestore.client()
        uid = str(user_id).strip()
        user_snap = db.collection("users").document(uid).get()
        tz_name = None
        if user_snap.exists:
            tz_name = (user_snap.to_dict() or {}).get("timeZone")

        day = date_str or _today_ymd_in_timezone(tz_name)
        snap = (
            db.collection("users")
            .document(uid)
            .collection("to-do-posts")
            .where("date", "==", day)
            .get()
        )
        todos: List[Dict[str, Any]] = []
        for doc in snap:
            row = doc.to_dict() or {}
            detail = row.get("detail")
            slice_detail = ""
            if isinstance(detail, str) and detail:
                slice_detail = _truncate(detail, MAX_DETAIL_CHARS)
            todos.append(
                {
                    "title": row.get("title"),
                    "detail": slice_detail,
                    "start": row.get("start"),
                    "completed": row.get("completed"),
                    "typeOfTodo": row.get("typeOfTodo"),
                    "todoID": row.get("todoID") or doc.id,
                    "date": row.get("date"),
                }
            )
        return todos
    except Exception as exc:
        logger.warning("fetch_today_todos_from_firestore failed for %s: %s", user_id, exc)
        return []


def build_personalization_for_request(
    data: Dict[str, Any],
    user_query: str,
) -> str:
    """
    Build a bounded personalization block for an LLM call.

    Accepts optional precomputed fields on ``data`` (identity_context, today_todos).
    When ``user_id`` is present, missing fields are loaded from Firestore + RAG.
    """
    user_id = data.get("user_id")
    if isinstance(user_id, str):
        user_id = user_id.strip() or None
    else:
        user_id = None

    identity_context = data.get("identity_context")
    if not identity_context and user_id:
        identity_context = fetch_identity_context_from_firestore(user_id)

    identity_block = ""
    if identity_context:
        try:
            from planner_utils import _format_identity_context

            identity_block = _format_identity_context(identity_context)
        except Exception as exc:
            logger.warning("identity formatting failed: %s", exc)

    today_todos = data.get("today_todos")
    if today_todos is None and user_id:
        today_todos = fetch_today_todos_from_firestore(user_id)
    if not isinstance(today_todos, list):
        today_todos = []

    active_plans = data.get("active_plans")
    if active_plans is None and user_id:
        active_plans = fetch_active_plans_from_firestore(user_id)
    if not isinstance(active_plans, list):
        active_plans = []
    plans_block = format_active_plans_block(active_plans)

    life_months_block = ""
    client_month_summary = data.get("month_life_summary")
    if client_month_summary:
        life_months_block = format_month_life_from_client(client_month_summary)
    anchor_ym = data.get("focus_year_month") or data.get("anchor_year_month")
    if not life_months_block and user_id:
        life_months_block = build_life_month_context_from_firestore(user_id, anchor_ym)

    current_todo = data.get("todo_data") if isinstance(data.get("todo_data"), dict) else {}
    current_id = _safe_todo_id(current_todo)

    today_block = format_today_todos_summary(
        today_todos,
        current_todo_id=current_id,
        max_items=MAX_TODAY_TODOS,
    )

    rag_block = ""
    if user_id and user_query:
        try:
            from user_memory import retrieve_user_context

            chunks = retrieve_user_context(user_id, user_query, top_k=MAX_RAG_CHUNKS)
            rag_block = format_rag_block(chunks)
        except Exception as exc:
            logger.debug("RAG retrieval skipped: %s", exc)

    intent_block = ""
    if user_id:
        intent_block = format_intent_profile_block(fetch_intent_profile_from_firestore(user_id))

    return assemble_personalization_block(
        identity_block=identity_block,
        life_months_block=life_months_block,
        plans_block=plans_block,
        today_block=today_block,
        rag_block=rag_block,
        intent_block=intent_block,
    )
