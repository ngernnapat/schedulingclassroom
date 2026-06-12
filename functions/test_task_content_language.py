#!/usr/bin/env python3
"""Unit tests for generate_task_content language + drill resolution.

These lock in the alignment fixes (Chinese-for-travel etc.) so a future prompt
edit cannot silently regress them. Pure functions only — no network/Firestore.

Run:
    OPENAI_API_KEY=test python test_task_content_language.py
    OPENAI_API_KEY=test pytest test_task_content_language.py
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test")  # avoid client init at import

import main as m


# --- practice-language detection -------------------------------------------

def test_chinese_detected_from_thai_planname():
    blob = "30-day การเรียนรู้ภาษาจีนเพื่อการท่องเที่ยว ฝึกออกเสียง"
    assert m._task_content_practice_language_from_hints(blob) == "Chinese"


def test_chinese_detected_from_hsk():
    assert m._task_content_practice_language_from_hints("hsk 1 เตรียมสอบ") == "Chinese"


def test_chinese_detected_from_mandarin_thai():
    assert m._task_content_practice_language_from_hints("เรียนแมนดาริน") == "Chinese"


def test_japanese_and_korean_detected():
    assert m._task_content_practice_language_from_hints("jlpt n5 คันจิ") == "Japanese"
    assert m._task_content_practice_language_from_hints("topik ฮันกึล") == "Korean"


def test_many_target_languages_detected():
    cases = {
        "เรียนภาษาฝรั่งเศส delf": "French",
        "ภาษาสเปนเพื่อการท่องเที่ยว": "Spanish",
        "learn italian for travel": "Italian",
        "ภาษาโปรตุเกสเบื้องต้น": "Portuguese",
        "เรียนภาษาเวียดนามเบื้องต้น": "Vietnamese",
        "ภาษาอินโดนีเซียสำหรับธุรกิจ": "Indonesian",
        "เรียนภาษารัสเซีย": "Russian",
        "ภาษาอาหรับเบื้องต้น": "Arabic",
        "ฝึกภาษาฮินดี": "Hindi",
    }
    for blob, want in cases.items():
        assert m._task_content_practice_language_from_hints(blob.lower()) == want, blob


def test_guard_works_across_scripts():
    # Cyrillic / Arabic / Devanagari targets are now script-checkable.
    assert not m._task_content_drill_uses_practice_language("privet — สวัสดี", "Russian")
    assert m._task_content_drill_uses_practice_language("привет (ปรีเวียต) — สวัสดี", "Russian")
    assert m._task_content_drill_uses_practice_language("مرحبا — hello", "Arabic")
    assert m._task_content_drill_uses_practice_language("नमस्ते — hello", "Hindi")
    # Latin-script targets remain unverifiable (assume fine, no false regen).
    assert m._task_content_needs_practice_script_check("Thai", "Italian") is False
    assert m._task_content_needs_practice_script_check("Thai", "Vietnamese") is False


def test_practice_language_resolves_chinese_for_food_step():
    plan = "30-Day การเรียนรู้ภาษาจีนเพื่อการท่องเที่ยว"
    step = "เรียนคำศัพท์ส่วนผสม 8 คำ เช่น เนื้อ ไก่ หมู ไข่ ผัก แล้วอ่านออกเสียง"
    instr = m._task_content_instruction_language(step, "", "thai")
    practice = m._task_content_practice_language(step, "", None, plan, "thai", "", instr)
    assert instr == "Thai"
    assert practice == "Chinese"


# --- instruction language ---------------------------------------------------

def test_instruction_language_thai_from_script():
    assert m._task_content_instruction_language("ฝึกออกเสียงพินอิน", "", "english") == "Thai"


def test_instruction_language_english():
    assert m._task_content_instruction_language("practice pinyin sounds", "", "english") == "English"


# --- coaching language (no English leak) ------------------------------------

def test_coaching_thai_for_thai_step_even_when_ui_english():
    # The bug: app default languageSelected=english leaked English coaching into
    # a Thai step. Coaching must follow the step's language.
    coach = m._task_content_coaching_language(
        "Thai", "English", "Chinese", "learning_language", "ฝึกออกเสียง", ""
    )
    assert coach == "Thai"


def test_coaching_english_for_english_step():
    coach = m._task_content_coaching_language(
        "English", "English", "Chinese", "learning_language", "practice tones", ""
    )
    assert coach == "English"


def test_coaching_immersive_english_quiz_for_english_plan_thai_ui():
    # The screenshot bug: English-learning plan, English step, Thai app UI.
    # The quiz must be in English (immersion), NOT translated to Thai.
    coach = m._task_content_coaching_language(
        "English", "Thai", "English", "learning_language",
        "Complete a short baseline assessment: 1 reading passage question set", "",
    )
    assert coach == "English"


def test_coaching_thai_for_chinese_drill_stays_thai():
    # Mirror case must still hold: Thai step drilling Chinese → Thai quiz.
    coach = m._task_content_coaching_language(
        "Thai", "Thai", "Chinese", "learning_language", "ฝึกออกเสียงพินอิน 5 คำ", ""
    )
    assert coach == "Thai"


def test_coaching_nonlanguage_english_step_thai_ui_stays_thai():
    # A fitness plan with an English step + Thai UI should still coach in Thai —
    # the user is not learning English, so no immersion.
    coach = m._task_content_coaching_language(
        "English", "Thai", "English", "exercise", "Do 3 sets of squats", ""
    )
    assert coach == "Thai"


# --- thin-step detection ----------------------------------------------------

def test_thin_steps_flagged():
    for s in ("ทบทวน", "ฝึก", "Review", "Practice speaking"):
        assert m._task_content_step_is_thin(s, ""), s


def test_specific_steps_not_thin():
    for s in ("ฝึกออกเสียงพินอิน 5 คำ", "Practice 6 pinyin initial sounds"):
        assert not m._task_content_step_is_thin(s, ""), s


# --- drill-sheet translate rule --------------------------------------------

def test_translate_rule_present_for_foreign_drill():
    out = m._task_content_drill_sheet_requirements(
        "เรียนคำศัพท์ เช่น เนื้อ ไก่", "ภาษาจีน", "Chinese", "Thai"
    )
    assert "TRANSLATE" in out
    assert "牛肉" in out  # worked example for Chinese


def test_translate_rule_absent_when_same_language():
    out = m._task_content_drill_sheet_requirements(
        "review vocab", "English plan", "English", "English"
    )
    assert "TRANSLATE" not in out


# --- output-quality guard (practice-script check) ---------------------------

def test_guard_flags_thai_copy():
    # The exact screenshot failure: Thai words drilled as if they were Chinese.
    bad = "1) เนื้อ — คำอ่าน: เนื้อ — ความหมาย: meat\n2) ไก่ — คำอ่าน: ไก่ — chicken"
    assert not m._task_content_drill_uses_practice_language(bad, "Chinese")


def test_guard_passes_real_chinese():
    good_hanzi = "牛肉 niúròu (หนิว โร่ว) — เนื้อ"
    good_pinyin = "niúròu (หนิว โร่ว) — เนื้อ"
    assert m._task_content_drill_uses_practice_language(good_hanzi, "Chinese")
    assert m._task_content_drill_uses_practice_language(good_pinyin, "Chinese")


def test_guard_skips_latin_target():
    # Can't reliably check latin-script targets — assume fine.
    assert m._task_content_needs_practice_script_check("Thai", "French") is False
    assert m._task_content_drill_uses_practice_language("bonjour — สวัสดี", "French")


def test_guard_needs_check_only_when_languages_differ():
    assert m._task_content_needs_practice_script_check("Thai", "Chinese") is True
    assert m._task_content_needs_practice_script_check("Chinese", "Chinese") is False


# --- thin-step directive is artifact-aware ----------------------------------

def test_thin_directive_card_phrasing():
    card = m._task_content_thin_step_directive(
        "ภาษาจีน", "ทบทวน", True, "Chinese", artifact="practice_card"
    )
    assert "scenario" in card.lower()
    drill = m._task_content_thin_step_directive(
        "ภาษาจีน", "ทบทวน", True, "Chinese", artifact="content"
    )
    assert "drill" in drill.lower()


# --- generate_practice shares the guard + has a cache version ---------------

def test_practice_format_version_bumped():
    assert m._PRACTICE_FORMAT_VERSION >= 2


def test_practice_card_guard_reuses_script_check():
    # Same guard powers both endpoints: a Thai-only "Chinese" card is rejected.
    thai_only = "วันนี้คุณอยู่ที่ร้านอาหาร สั่งอาหารเป็นภาษาไทย"
    assert not m._task_content_drill_uses_practice_language(thai_only, "Chinese")
    with_chinese = "ที่ร้าน พูดว่า 你好 (หนี่ห่าว) แล้วสั่ง 牛肉面"
    assert m._task_content_drill_uses_practice_language(with_chinese, "Chinese")


# --- minimal runner (no pytest dependency) ----------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}  {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}  {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
