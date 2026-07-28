"""plan_time — the one place a plan task's clock time is normalized.

EVERY mobile reader of a task's `time` (the day timeline block in aiModules,
the day's own calendar start time in evoRealtimeTaskActions, the timeline
editor) tests it against `^\\d{1,2}:\\d{2}$` and silently DROPS whatever fails.
So a model that writes `9:00 AM`, `9.00` or `09:00-11:00` does not produce a
slightly-off itinerary — it produces a day with no schedule at all, and the
plan quietly loses the thing that made it a timeline.

Both producers go through here: AI generation (generate_planner_content.Task)
and itinerary image import (main._sanitize_plan_image_import). Deliberately a
standalone module with no dependency beyond `re`, so main.py can import it at
module level without paying the generator's cold start.
"""

import re
from typing import Optional

_HHMM_RE = re.compile(r"(\d{1,2})\s*[:.]\s*(\d{2})")
# Letter lookarounds rather than \b: `\bam` never matches "9am" (digit→letter is
# not a word boundary), while a bare `am` would match inside "amenities".
_PM_RE = re.compile(r"(?<![a-z])p\.?m\.?(?![a-z])")
_AM_RE = re.compile(r"(?<![a-z])a\.?m\.?(?![a-z])")


def normalize_clock_time(value: object) -> Optional[str]:
    """Any clock time a model or a photographed itinerary writes → `HH:MM`, or None.

    Returns None rather than guessing: an out-of-range hour, or a bare number
    with nothing to disambiguate it ("3"), is not a time this can recover, and
    inventing one would put a real task at a made-up hour.
    """
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    meridiem = "pm" if _PM_RE.search(lowered) else ("am" if _AM_RE.search(lowered) else "")

    # A range ("09:00-11:00", "9-10am") starts when its FIRST time starts.
    match = _HHMM_RE.search(lowered)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
    else:
        # A bare hour is only readable when a meridiem pins it down.
        bare = re.search(r"\d{1,2}", lowered) if meridiem else None
        if not bare:
            return None
        hour, minute = int(bare.group(0)), 0

    if minute > 59:
        return None
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return f"{hour:02d}:{minute:02d}"
