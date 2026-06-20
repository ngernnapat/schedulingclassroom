"""On-demand context tools for EVO voice chat — fetch calendar/plans only when asked."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

MAX_VOICE_CALENDAR_ITEMS = 10
MAX_TITLE_CHARS = 48
MAX_DETAIL_CHARS = 80

# Cache resolved timezones per uid for the life of the (warm) instance so we don't
# re-read the user doc on every calendar tool call within a turn.
_TZ_CACHE: Dict[str, "ZoneInfo"] = {}


def _evo_db():
    try:
        from evo_firebase import evo_firestore

        db = evo_firestore()
        if db is not None:
            return db
        logger.warning(
            "EVO Firestore unavailable for voice tools — set EVO_FIREBASE_SERVICE_ACCOUNT_JSON"
        )
    except Exception as exc:
        logger.warning("EVO Firestore unavailable for voice tools: %s", exc)
    return None


def _user_timezone(user_id: str) -> ZoneInfo:
    uid = str(user_id or "").strip()
    if uid in _TZ_CACHE:
        return _TZ_CACHE[uid]
    db = _evo_db()
    tz_name = None
    if db and uid:
        try:
            snap = db.collection("users").document(uid).get()
            if snap.exists:
                tz_name = (snap.to_dict() or {}).get("timeZone")
        except Exception:
            pass
    try:
        tz = ZoneInfo(tz_name) if tz_name else ZoneInfo("Asia/Bangkok")
    except Exception:
        tz = ZoneInfo("Asia/Bangkok")
    if uid:
        _TZ_CACHE[uid] = tz
    return tz


def _date_for_offset(user_id: str, day_offset: int) -> str:
    tz = _user_timezone(user_id)
    day = datetime.now(tz).date() + timedelta(days=day_offset)
    return day.strftime("%Y-%m-%d")


def _truncate(text: Any, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _row_matches_date(row: Dict[str, Any], date_str: str, tz: ZoneInfo) -> bool:
    raw_date = row.get("date")
    if isinstance(raw_date, str) and raw_date.strip()[:10] == date_str:
        return True
    fds = row.get("fullDateStamp")
    if fds is None:
        return False
    try:
        if hasattr(fds, "timestamp"):
            dt = datetime.fromtimestamp(fds.timestamp(), tz=tz)
        elif isinstance(fds, datetime):
            dt = fds.astimezone(tz) if fds.tzinfo else fds.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        else:
            return False
        return dt.strftime("%Y-%m-%d") == date_str
    except Exception:
        return False


def _normalize_voice_todo(row: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    return {
        "title": row.get("title"),
        "detail": _truncate(row.get("detail"), MAX_DETAIL_CHARS),
        "start": row.get("start"),
        "end": row.get("end"),
        "completed": row.get("completed") if row.get("completed") is not None else row.get("isCompleted"),
        "typeOfTodo": row.get("typeOfTodo"),
        "todoID": row.get("todoID") or doc_id,
        "date": row.get("date"),
        "planName": row.get("planName"),
    }


def fetch_todos_for_date(user_id: str, date_str: str) -> List[Dict[str, Any]]:
    if not user_id or not date_str:
        return []
    db = _evo_db()
    if not db:
        return []
    uid = str(user_id).strip()
    tz = _user_timezone(uid)
    todos: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _add(doc) -> None:
        row = doc.to_dict() or {}
        key = str(row.get("todoID") or doc.id)
        if key in seen:
            return
        seen.add(key)
        todos.append(_normalize_voice_todo(row, doc.id))

    try:
        snap = (
            db.collection("users")
            .document(uid)
            .collection("to-do-posts")
            .where("date", "==", date_str)
            .get()
        )
        for doc in snap:
            _add(doc)

        if not todos and len(date_str) >= 7:
            ym = date_str[:7]
            month_snap = (
                db.collection("users")
                .document(uid)
                .collection("to-do-posts")
                .where("yearMonthStamp", "==", ym)
                .get()
            )
            for doc in month_snap:
                row = doc.to_dict() or {}
                if _row_matches_date(row, date_str, tz):
                    _add(doc)
    except Exception as exc:
        logger.warning("voice_tools fetch_todos_for_date failed for %s: %s", uid, exc)
    return todos


def fetch_todos_range(user_id: str, start_ymd: str, end_ymd: str) -> List[Dict[str, Any]]:
    """Fetch all todos in an inclusive YYYY-MM-DD range with ONE Firestore query.

    The `date` field is stored as YYYY-MM-DD, which sorts lexicographically, so a
    single range query covers a whole week/month instead of one query per day.
    """
    if not user_id or not start_ymd or not end_ymd:
        return []
    db = _evo_db()
    if not db:
        return []
    uid = str(user_id).strip()
    todos: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    try:
        snap = (
            db.collection("users")
            .document(uid)
            .collection("to-do-posts")
            .where("date", ">=", start_ymd)
            .where("date", "<=", end_ymd)
            .get()
        )
        for doc in snap:
            row = doc.to_dict() or {}
            key = str(row.get("todoID") or doc.id)
            if key in seen:
                continue
            seen.add(key)
            todos.append(_normalize_voice_todo(row, doc.id))
    except Exception as exc:
        logger.warning("voice_tools fetch_todos_range failed for %s: %s", uid, exc)
    return todos


def _sort_todos(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(task: Dict[str, Any]) -> tuple:
        start = str(task.get("start") or "").strip()
        if not start:
            return (1, 9999)
        parts = start.split(":")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1][:2].isdigit():
            return (0, int(parts[0]) * 60 + int(parts[1][:2]))
        return (0, start)

    return sorted([t for t in todos if isinstance(t, dict)], key=_key)


def _format_todo_line(task: Dict[str, Any]) -> str:
    title = _truncate(task.get("title") or "Task", MAX_TITLE_CHARS)
    start = str(task.get("start") or "").strip()
    done = task.get("completed") is True
    mark = "✓ " if done else ""
    line = f"• {mark}{title}"
    if start:
        line += f" ({start})"
    plan = str(task.get("planName") or "").strip()
    if plan:
        line += f" [{_truncate(plan, 20)}]"
    return line


def format_day_schedule(todos: List[Dict[str, Any]], label: str, date_str: str) -> str:
    active = _sort_todos(todos)
    if not active:
        return f"{label} ({date_str}): no calendar items."
    lines = [_format_todo_line(t) for t in active[:MAX_VOICE_CALENDAR_ITEMS]]
    remaining = len(active) - len(lines)
    if remaining > 0:
        lines.append(f"• +{remaining} more")
    return f"{label} ({date_str}, {len(active)} items):\n" + "\n".join(lines)


def _combined_utterance(user_text: str, chat_history: Optional[List[Dict[str, Any]]]) -> str:
    parts: List[str] = [str(user_text or "").strip()]
    if isinstance(chat_history, list):
        for msg in chat_history[-4:]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = msg.get("text")
            if role == "user" and isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return " ".join(parts).lower()


def detect_voice_tool_requests(
    user_text: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Set[str]:
    """Return tool ids to run based on what the user is asking about."""
    blob = _combined_utterance(user_text, chat_history)
    if not blob.strip():
        return set()

    tools: Set[str] = set()

    day_rules: List[Tuple[str, int, List[str]]] = [
        (
            "calendar_today",
            0,
            [
                r"\btoday\b",
                r"\btoday'?s\b",
                r"this morning",
                r"this afternoon",
                r"tonight",
                r"วันนี้",
                r"ตอนนี้",
                r"งานวันนี้",
                r"todo.*today",
                r"task.*today",
                r"tasks today",
                r"todos today",
            ],
        ),
        (
            "calendar_tomorrow",
            1,
            [
                r"\btomorrow\b",
                r"tomorrow morning",
                r"tomorrow'?s",
                r"tasks? for tomorrow",
                r"todo.*tomorrow",
                r"tomorrow tasks",
                r"พรุ่งนี้",
                r"วันพรุ่ง",
                r"งานพรุ่งนี้",
                r"พรุ่งนี้มี",
            ],
        ),
        (
            "calendar_yesterday",
            -1,
            [r"\byesterday\b", r"เมื่อวาน"],
        ),
    ]

    for tool_id, _offset, patterns in day_rules:
        if any(re.search(p, blob, re.I) for p in patterns):
            tools.add(tool_id)

    if re.search(
        r"this week|week ahead|next week|weekly schedule|tasks? this week|"
        r"this week'?s tasks|สัปดาห์นี้|สัปดาห์หน้า|อาทิตย์นี้|งานสัปดาห์นี้",
        blob,
        re.I,
    ):
        tools.add("calendar_week")

    if re.search(r"this month|month ahead|whole month|all month|เดือนนี้|เดือนหน้า|ทั้งเดือน", blob, re.I):
        tools.add("calendar_month")

    calendar_generic = re.search(
        r"calendar|schedule|agenda|my day|what'?s on|what do i have|"
        r"any (meetings|tasks|plans)|ปฏิทิน|ตาราง|มีอะไร|งานวัน",
        blob,
        re.I,
    )
    if calendar_generic and not any(t.startswith("calendar_") for t in tools):
        tools.add("calendar_today")

    if re.search(
        r"\bplan\b|lifestyle plan|active plan|my program|แผน|โปรแกรม|"
        r"plan day|day \d+ of",
        blob,
        re.I,
    ):
        tools.add("active_plans")

    if re.search(
        r"profile|about me|mbti|streak|โปรไฟล",
        blob,
        re.I,
    ):
        tools.add("profile")

    if re.search(
        r"life goal|my goal|purpose|meaning|want to (?:be|become)|who i want to be|"
        r"my why|aspiration|identity|north star|เป้าหมาย(?:ชีวิต)?|ความหมาย(?:ของชีวิต)?|"
        r"อยากเป็น|ตัวตน|แรงบันดาลใจ",
        blob,
        re.I,
    ):
        tools.add("goals")

    if re.search(
        r"how am i doing|how'?s it going|my progress|on track|adherence|"
        r"review|overview|summary|recap|completed|how many.*(done|left)|"
        r"ความคืบหน้า|เป็นยังไงบ้าง|สรุป|รีวิว|ทำไปกี่|เหลือกี่",
        blob,
        re.I,
    ):
        tools.add("planner_overview")

    if re.search(
        r"what should i (do|focus|start)|focus on|most important|priorit|"
        r"where (do i|to) start|what'?s first|what do i do first|"
        r"จัดลำดับ|อะไรสำคัญ|ทำอะไรก่อน|เริ่มตรงไหน|โฟกัส",
        blob,
        re.I,
    ):
        tools.add("prioritize")

    if re.search(
        r"\bremember\b|\brecall\b|do you (know|remember)|you know that|what do you know about me|"
        r"my notes|my favou?rite|จำได้|เคยบอก|รู้จักฉัน|ที่เคยบอก|ที่ชอบ",
        blob,
        re.I,
    ):
        tools.add("user_notes")

    return tools


def _tool_calendar_day(user_id: str, day_offset: int, label: str) -> str:
    date_str = _date_for_offset(user_id, day_offset)
    todos = fetch_todos_for_date(user_id, date_str)
    return format_day_schedule(todos, label, date_str)


def _tool_calendar_week(user_id: str, is_thai: bool, data: Optional[Dict[str, Any]] = None) -> str:
    uid = str(user_id).strip()
    if not uid:
        return ""
    tz = _user_timezone(uid)
    start = datetime.now(tz).date()
    end = start + timedelta(days=6)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    empty = "สัปดาห์นี้: ไม่มีรายการใน 7 วันข้างหน้า" if is_thai else "This week: no calendar items in the next 7 days."
    server_todos = fetch_todos_range(uid, start_str, end_str)
    client_todos = [
        t
        for t in _client_calendar_todos(data)
        if start_str <= str(t.get("date") or "")[:10] <= end_str
    ]
    todos = _pick_todos(server_todos, client_todos)
    if not todos:
        return empty

    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for task in todos:
        day_key = str(task.get("date") or "")[:10]
        if day_key:
            by_date.setdefault(day_key, []).append(task)

    lines: List[str] = []
    for offset in range(7):
        day = start + timedelta(days=offset)
        day_todos = by_date.get(day.strftime("%Y-%m-%d"))
        if not day_todos:
            continue
        if offset == 0:
            day_label = "วันนี้" if is_thai else "Today"
        elif offset == 1:
            day_label = "พรุ่งนี้" if is_thai else "Tomorrow"
        else:
            day_label = day.strftime("%a %b %d")
        pending = sum(1 for t in day_todos if t.get("completed") is not True)
        titles = ", ".join(_truncate(t.get("title"), 24) for t in _sort_todos(day_todos)[:3])
        lines.append(f"{day_label}: {pending} item(s) — {titles}")
    if not lines:
        return empty
    header = "สัปดาห์นี้:" if is_thai else "Week ahead:"
    return header + "\n" + "\n".join(f"• {ln}" for ln in lines)


def _tool_calendar_month(user_id: str, is_thai: bool) -> str:
    uid = str(user_id).strip()
    if not uid:
        return ""
    tz = _user_timezone(uid)
    today = datetime.now(tz).date()
    first = today.replace(day=1)
    nxt = first.replace(year=first.year + 1, month=1) if first.month == 12 \
        else first.replace(month=first.month + 1)
    last = nxt - timedelta(days=1)
    ym = today.strftime("%Y-%m")
    todos = fetch_todos_range(uid, first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
    if not todos:
        return f"เดือนนี้ ({ym}): ไม่มีรายการ" if is_thai else f"This month ({ym}): no calendar items."

    total = len(todos)
    done = sum(1 for t in todos if t.get("completed") is True)
    today_str = today.strftime("%Y-%m-%d")
    upcoming = sorted(
        (t for t in todos if str(t.get("date") or "")[:10] >= today_str),
        key=lambda t: (str(t.get("date") or ""), str(t.get("start") or "")),
    )
    header = (
        f"เดือนนี้ ({ym}): {total} รายการ, เสร็จแล้ว {done}"
        if is_thai
        else f"This month ({ym}): {total} items, {done} done"
    )
    lines = [f"• {str(t.get('date') or '')[5:10]} {_truncate(t.get('title'), 28)}" for t in upcoming[:8]]
    if not lines:
        return header
    sub = "ที่จะถึง:" if is_thai else "Upcoming:"
    return f"{header}\n{sub}\n" + "\n".join(lines)


def _planner_period_range(period: str, tz: ZoneInfo):
    today = datetime.now(tz).date()
    if period == "day":
        return today, today
    if period == "month":
        first = today.replace(day=1)
        nxt = first.replace(year=first.year + 1, month=1) if first.month == 12 \
            else first.replace(month=first.month + 1)
        return first, nxt - timedelta(days=1)
    return today, today + timedelta(days=6)  # week


def _tool_planner_overview(user_id: str, period: str, is_thai: bool) -> str:
    """Adherence overview for a period: tasks done vs planned, broken down by plan."""
    uid = str(user_id).strip()
    if not uid:
        return ""
    tz = _user_timezone(uid)
    start, end = _planner_period_range(period, tz)
    todos = fetch_todos_range(uid, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    label = {
        "day": "วันนี้" if is_thai else "Today",
        "week": "สัปดาห์นี้" if is_thai else "This week",
        "month": "เดือนนี้" if is_thai else "This month",
    }.get(period, period)

    total = len(todos)
    if not total:
        return f"{label}: " + ("ยังไม่มีงานในแพลนเนอร์" if is_thai else "no planner tasks scheduled.")

    done = sum(1 for t in todos if t.get("completed") is True)

    plans: Dict[str, Dict[str, int]] = {}
    for t in todos:
        name = str(t.get("planName") or "").strip()
        if not name:
            continue
        bucket = plans.setdefault(name, {"count": 0, "done": 0})
        bucket["count"] += 1
        if t.get("completed") is True:
            bucket["done"] += 1

    head = (
        f"{label}: เสร็จ {done}/{total} งาน"
        if is_thai
        else f"{label}: {done}/{total} tasks done"
    )
    lines = [head]
    for name, s in list(plans.items())[:4]:
        lines.append(f"• {_truncate(name, 32)}: {s['done']}/{s['count']}")
    return "\n".join(lines)


def _priority_range(period: str, tz: ZoneInfo):
    """Includes a 7-day lookback so overdue reps surface for prioritization."""
    today = datetime.now(tz).date()
    if period == "month":
        first = today.replace(day=1)
        nxt = first.replace(year=first.year + 1, month=1) if first.month == 12 \
            else first.replace(month=first.month + 1)
        return first, nxt - timedelta(days=1)
    if period == "week":
        return today - timedelta(days=7), today + timedelta(days=6)
    return today - timedelta(days=7), today  # day


def _score_task(task: Dict[str, Any], today_str: str) -> Tuple[int, str]:
    date = str(task.get("date") or "")[:10]
    plan_linked = bool(task.get("planName"))
    overdue = bool(date) and date < today_str
    is_today = date == today_str

    points = 0
    if plan_linked:
        points += 30
    if overdue:
        points += 40
        if plan_linked:
            points += 15
    elif is_today:
        points += 20 if task.get("start") else 10
    else:
        points += 5

    if overdue:
        reason = "overdue — a recovery rep"
    elif plan_linked:
        reason = f"part of {_truncate(task.get('planName'), 24)}"
    elif is_today and task.get("start"):
        reason = f"set for {task.get('start')}"
    elif is_today:
        reason = "scheduled today"
    else:
        reason = "coming up"
    return points, reason


def _tool_prioritize(user_id: str, period: str, is_thai: bool) -> str:
    uid = str(user_id).strip()
    if not uid:
        return ""
    tz = _user_timezone(uid)
    start, end = _priority_range(period, tz)
    today_str = datetime.now(tz).strftime("%Y-%m-%d")
    todos = fetch_todos_range(uid, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    pending = [t for t in todos if t.get("completed") is not True]

    if not pending:
        return "ไม่มีงานค้างให้จัดลำดับ" if is_thai else "No open tasks to prioritize right now."

    ranked = sorted(
        ((_score_task(t, today_str), t) for t in pending),
        key=lambda x: -x[0][0],
    )
    focus = ranked[:3]
    optional = ranked[3:]

    header = "ลำดับโฟกัส:" if is_thai else "Focus order:"
    lines = [header]
    for idx, ((_pts, reason), t) in enumerate(focus, 1):
        lines.append(f"{idx}. {_truncate(t.get('title'), 40)} — {reason}")
    if optional:
        note = (
            f"งานเยอะ ({len(pending)} ค้าง) — โฟกัส {len(focus)} อย่างนี้ก่อน อีก {len(optional)} เลื่อนหรือพักได้"
            if is_thai
            else f"Heavy load ({len(pending)} open) — do these {len(focus)} first; the other {len(optional)} can wait or rest."
        )
        lines.append(note)
    return "\n".join(lines)


def _tool_active_plans(user_id: str) -> str:
    db = _evo_db()
    if not db or not user_id:
        return "No active lifestyle plans."
    uid = str(user_id).strip()
    try:
        snap = db.collection("users").document(uid).collection("lifestyle-plans").get()
        plans: List[Dict[str, Any]] = []
        for doc in snap:
            row = doc.to_dict() or {}
            if row.get("isDraft") is True:
                continue
            plans.append(
                {
                    "planName": row.get("planName") or "",
                    "detailPrompt": row.get("detailPrompt") or row.get("detail") or "",
                    "category": row.get("category") or "",
                }
            )
        if not plans:
            return "No active lifestyle plans."
        lines = []
        for plan in plans[:3]:
            name = _truncate(plan.get("planName") or "Plan", MAX_TITLE_CHARS)
            detail = _truncate(plan.get("detailPrompt"), MAX_DETAIL_CHARS)
            line = f"• {name}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        if len(plans) > 3:
            lines.append(f"• +{len(plans) - 3} more")
        return "Active lifestyle plans:\n" + "\n".join(lines)
    except Exception as exc:
        logger.warning("voice_tools active_plans failed: %s", exc)
        return ""


def _tool_profile(user_id: str) -> str:
    db = _evo_db()
    if not db or not user_id:
        return ""
    uid = str(user_id).strip()
    try:
        snap = (
            db.collection("users")
            .document(uid)
            .collection("identityProfile")
            .document("current")
            .get()
        )
        if not snap.exists:
            return "Profile: no streak data yet."
        profile = snap.to_dict() or {}
        streak = profile.get("currentStreak") or 0
        longest = profile.get("longestStreak") or 0
        badge = profile.get("latestBadge")
        parts = [f"Current streak: {streak} days", f"Longest streak: {longest} days"]
        if badge:
            parts.append(f"Latest badge: {badge}")
        return "Profile:\n" + "\n".join(f"• {p}" for p in parts)
    except Exception as exc:
        logger.warning("voice_tools profile failed: %s", exc)
        return ""


def _tool_goals(user_id: str, is_thai: bool) -> str:
    """The user's north star — life goal + identity context to ground coaching."""
    db = _evo_db()
    if not db or not user_id:
        return ""
    uid = str(user_id).strip()
    try:
        snap = db.collection("users").document(uid).get()
        u = (snap.to_dict() or {}) if snap.exists else {}
    except Exception as exc:
        logger.warning("voice_tools goals failed: %s", exc)
        return ""

    life = str(u.get("lifeGoals") or u.get("coachLifeGoals") or "").strip()
    work = str(u.get("currentWork") or u.get("occupation") or u.get("jobTitle") or "").strip()
    mbti = str(u.get("mbti") or u.get("mbtiType") or "").strip()

    if not (life or work or mbti):
        return (
            "เป้าหมายชีวิต: ยังไม่ได้ตั้ง — ชวนผู้ใช้คุยว่าอยากเป็นอะไร แล้วเสนอบันทึกไว้"
            if is_thai
            else "Life goal: not set yet — invite the user to share what they want to become, and offer to save it."
        )

    parts = []
    if life:
        parts.append(f"เป้าหมายชีวิต: {life}" if is_thai else f"Life goal: {life}")
    if work:
        parts.append(f"งาน: {work}" if is_thai else f"Work: {work}")
    if mbti:
        parts.append(f"MBTI: {mbti}")
    header = "north star ของผู้ใช้ (เชื่อมคำแนะนำกับสิ่งนี้):" if is_thai else "User's north star (connect advice to this):"
    return header + "\n" + "\n".join(f"• {p}" for p in parts)


def _tool_user_notes(user_id: str, is_thai: bool) -> str:
    """Recall freeform coaching notes (preferences/facts) saved about the user."""
    db = _evo_db()
    if not db or not user_id:
        return ""
    uid = str(user_id).strip()
    try:
        snap = db.collection("users").document(uid).collection("coachNotes").limit(50).get()
        notes = []
        for doc in snap:
            row = doc.to_dict() or {}
            text = str(row.get("text") or "").strip()
            if text:
                notes.append(text)
    except Exception as exc:
        logger.warning("voice_tools user_notes failed: %s", exc)
        return ""
    if not notes:
        return ""
    header = "สิ่งที่จำเกี่ยวกับผู้ใช้:" if is_thai else "What you remember about the user:"
    return header + "\n" + "\n".join(f"• {_truncate(n, 80)}" for n in notes[:12])


def _client_calendar_todos(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    raw = data.get("calendar_todos") or data.get("calendarTodos") or []
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, dict)]


def _pick_todos(server: List[Dict[str, Any]], client: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(client) > len(server):
        return client
    return server or client


def _todos_for_date(user_id: str, date_str: str, data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    server = fetch_todos_for_date(user_id, date_str)
    client = [
        t
        for t in _client_calendar_todos(data)
        if str(t.get("date") or "")[:10] == date_str
    ]
    today_client = data.get("today_todos") or data.get("todayTodos") if isinstance(data, dict) else []
    if isinstance(today_client, list) and date_str == _date_for_offset(user_id, 0):
        client = _pick_todos(client, [t for t in today_client if isinstance(t, dict)])
    return _pick_todos(server, client)


def build_voice_tool_context(
    user_id: str,
    user_text: str,
    chat_history: Optional[List[Dict[str, Any]]],
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Run only the tools implied by the user's question; return a compact text block."""
    if not user_id or not str(user_id).strip():
        return ""

    data = data or {}
    is_thai = str(data.get("language") or "").lower() in {"thai", "th", "ไทย"}
    tool_ids = detect_voice_tool_requests(user_text, chat_history)
    if not tool_ids:
        return ""

    blocks: List[str] = []
    uid = str(user_id).strip()

    if "calendar_today" in tool_ids:
        label = "วันนี้" if is_thai else "Today"
        date_str = _date_for_offset(uid, 0)
        todos = _todos_for_date(uid, date_str, data)
        if todos:
            logger.info("voice calendar_today %s items for %s", len(todos), uid)
        blocks.append(format_day_schedule(todos, label, date_str))
    if "calendar_tomorrow" in tool_ids:
        label = "พรุ่งนี้" if is_thai else "Tomorrow"
        date_str = _date_for_offset(uid, 1)
        todos = _todos_for_date(uid, date_str, data)
        blocks.append(format_day_schedule(todos, label, date_str))
    if "calendar_yesterday" in tool_ids:
        label = "เมื่อวาน" if is_thai else "Yesterday"
        date_str = _date_for_offset(uid, -1)
        todos = _todos_for_date(uid, date_str, data)
        blocks.append(format_day_schedule(todos, label, date_str))
    if "calendar_week" in tool_ids:
        week_block = _tool_calendar_week(uid, is_thai, data)
        if week_block:
            blocks.append(week_block)
    if "calendar_month" in tool_ids:
        month_block = _tool_calendar_month(uid, is_thai)
        if month_block:
            blocks.append(month_block)
    if "active_plans" in tool_ids:
        plan_block = _tool_active_plans(uid)
        if plan_block:
            blocks.append(plan_block)
    if "profile" in tool_ids:
        profile_block = _tool_profile(uid)
        if profile_block:
            blocks.append(profile_block)
    if "goals" in tool_ids:
        goals_block = _tool_goals(uid, is_thai)
        if goals_block:
            blocks.append(goals_block)
    if "user_notes" in tool_ids:
        notes_block = _tool_user_notes(uid, is_thai)
        if notes_block:
            blocks.append(notes_block)
    if "planner_overview" in tool_ids:
        blob = _combined_utterance(user_text, chat_history)
        if re.search(r"month|เดือน", blob, re.I):
            period = "month"
        elif re.search(r"today|tonight|วันนี้", blob, re.I):
            period = "day"
        else:
            period = "week"
        overview = _tool_planner_overview(uid, period, is_thai)
        if overview:
            blocks.append(overview)
    if "prioritize" in tool_ids:
        blob = _combined_utterance(user_text, chat_history)
        if re.search(r"month|เดือน", blob, re.I):
            p_period = "month"
        elif re.search(r"week|สัปดาห์|อาทิตย์", blob, re.I):
            p_period = "week"
        else:
            p_period = "day"
        ranked = _tool_prioritize(uid, p_period, is_thai)
        if ranked:
            blocks.append(ranked)

    if not blocks:
        return ""
    return "Fetched for this question:\n" + "\n\n".join(blocks)
