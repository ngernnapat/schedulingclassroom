# Compact Elder Futhark copy for LLM prompts (English). Mirrors EVOforluanching/components/elderFutharkRunes.js.
# Used when clients send only {key, becomingPhrase}; server fills name/meaning/category.

from typing import Any, Dict, List, Optional

RUNE_LLM_CATALOG: Dict[str, Dict[str, str]] = {
    "fehu": {"name": "Fehu", "meaning": "Wealth, mobile resources", "category": "growth", "becoming": "You're becoming someone who completes what they begin."},
    "uruz": {"name": "Uruz", "meaning": "Strength of the wild ox", "category": "consistency", "becoming": "You're proving you can hold a pattern. Strength is repetition, not perfection."},
    "thurisaz": {"name": "Thurisaz", "meaning": "Thorn — break through resistance", "category": "resilience", "becoming": "You came back. Returning is the rep that builds the brain you want."},
    "ansuz": {"name": "Ansuz", "meaning": "Inspired speech, signal from the source", "category": "wisdom", "becoming": "You found the words for who you're becoming."},
    "raido": {"name": "Raido", "meaning": "The journey, motion with direction", "category": "consistency", "becoming": "Twenty-one repetitions in. The path isn't theoretical anymore."},
    "kenaz": {"name": "Kenaz", "meaning": "Torch — knowledge made visible", "category": "growth", "becoming": "You picked a plan and lit the torch. The path is yours now."},
    "gebo": {"name": "Gebo", "meaning": "Gift — the bond of giving and receiving", "category": "community", "becoming": "Progress isn't yours alone anymore. Sharing is part of who you are."},
    "wunjo": {"name": "Wunjo", "meaning": "Joy — alignment with what you do", "category": "consistency", "becoming": "Fifty repetitions. Your brain is choosing this on its own now."},
    "hagalaz": {"name": "Hagalaz", "meaning": "Hail — disruption that reveals strength", "category": "resilience", "becoming": "Life broke the pattern, and you rebuilt it. That's the rarer skill."},
    "naudhiz": {"name": "Naudhiz", "meaning": "Need — the friction that forges discipline", "category": "resilience", "becoming": "You met yourself where you were and chose to keep going anyway."},
    "isa": {"name": "Isa", "meaning": "Ice — stillness, clarity, focus", "category": "mastery", "becoming": "Five out of seven days, no all-or-nothing. That's how plasticity actually works."},
    "jera": {"name": "Jera", "meaning": "Year, harvest — patience earning fruit", "category": "mastery", "becoming": "You stayed long enough to harvest what you planted."},
    "eihwaz": {"name": "Eihwaz", "meaning": "Yew tree — endurance, deep roots", "category": "consistency", "becoming": "One hundred and twenty repetitions. Your roots are in the practice itself."},
    "perthro": {"name": "Perthro", "meaning": "Lot-cup — engaging with chance and choice", "category": "wisdom", "becoming": "You let the AI guide a plan and stayed open to where it led."},
    "algiz": {"name": "Algiz", "meaning": "Elk — protection, sacred boundary", "category": "wisdom", "becoming": "You shared on your terms. Privacy is a form of self-respect."},
    "sowilo": {"name": "Sowilo", "meaning": "Sun — wholeness, victory, life force", "category": "consistency", "becoming": "Two hundred and fifty repetitions. The cortical wiring is different."},
    "tiwaz": {"name": "Tiwaz", "meaning": "Tyr — sacrifice for justice, leadership", "category": "community", "becoming": "Someone followed your lead. That's a different kind of strength."},
    "berkano": {"name": "Berkano", "meaning": "Birch — growth, gentle beginnings, regeneration", "category": "growth", "becoming": "You let yourself start lighter. That's how plants grow back."},
    "ehwaz": {"name": "Ehwaz", "meaning": "Horse — partnership, harmonious motion", "category": "growth", "becoming": "You moved between plans without losing rhythm."},
    "mannaz": {"name": "Mannaz", "meaning": "Self — the human, mirror of community", "category": "wisdom", "becoming": "You spoke about yourself in the language of who you're becoming."},
    "laguz": {"name": "Laguz", "meaning": "Water — flow, the unconscious, automatic motion", "category": "mastery", "becoming": "It happens without effort now. That's how you know it's yours."},
    "ingwaz": {"name": "Ingwaz", "meaning": "Ing — completion of a cycle, fertility of finished work", "category": "mastery", "becoming": "You finished a full plan. Not a habit — a completed arc."},
    "dagaz": {"name": "Dagaz", "meaning": "Day — breakthrough, the moment things flip", "category": "growth", "becoming": "Something clicked. The morning feels different now."},
    "othala": {"name": "Othala", "meaning": "Heritage — what you pass on", "category": "community", "becoming": "Your way became someone else's path forward."},
}


def normalize_earned_runes_for_llm(rows: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """Merge partial Firestore rows (key + optional becoming) with RUNE_LLM_CATALOG."""
    out: List[Dict[str, str]] = []
    if not rows:
        return out
    for raw in rows[:24]:
        if not isinstance(raw, dict):
            continue
        key = (raw.get("key") or raw.get("rune_key") or "").strip().lower()
        if not key:
            continue
        cat = RUNE_LLM_CATALOG.get(key, {})
        becoming = (
            raw.get("becoming")
            or raw.get("becomingPhrase")
            or cat.get("becoming")
            or ""
        )
        out.append(
            {
                "key": key,
                "name": str(raw.get("name") or cat.get("name") or key),
                "meaning": str(raw.get("meaning") or cat.get("meaning") or ""),
                "category": str(raw.get("category") or cat.get("category") or ""),
                "becoming": str(becoming),
            }
        )
    return out
