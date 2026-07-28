"""Focused tests for realtime voice capability packing."""

import unittest

from realtime_capabilities import (
    CORE_TOOL_NAMES,
    PACK_TOOL_NAMES,
    build_realtime_capability_payload,
    coach_chat_capability_payload,
    compact_realtime_instructions,
    loaded_packs_from_history,
    uncategorized_tool_names,
)


def _tool(name):
    return {
        "type": "function",
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }


class RealtimeCapabilitiesTest(unittest.TestCase):
    def setUp(self):
        names = set(CORE_TOOL_NAMES)
        for pack_names in PACK_TOOL_NAMES.values():
            names.update(pack_names)
        self.tools = [_tool(name) for name in sorted(names)]

    def test_every_business_tool_is_categorized(self):
        self.assertEqual(uncategorized_tool_names(self.tools), [])

    def test_initial_surface_is_core_plus_loader(self):
        initial, packs = build_realtime_capability_payload(self.tools, is_host=True)
        initial_names = [tool["name"] for tool in initial]
        self.assertEqual(set(initial_names[:-1]), set(CORE_TOOL_NAMES))
        self.assertEqual(initial_names[-1], "load_capability_pack")
        self.assertIn("planning", packs)
        self.assertIn("host", packs)

    def test_non_host_never_receives_host_pack(self):
        initial, packs = build_realtime_capability_payload(self.tools, is_host=False)
        self.assertNotIn("host", packs)
        loader = initial[-1]
        choices = loader["parameters"]["properties"]["pack"]["enum"]
        self.assertNotIn("host", choices)

    def test_compact_prompt_keeps_core_safety_and_language(self):
        prompt = compact_realtime_instructions(
            True,
            today_str="2026-07-24",
            tz_label="Asia/Bangkok",
            now_time="14:30",
        )
        self.assertIn("load_capability_pack", prompt)
        self.assertIn("two-phase", prompt)
        self.assertIn("Reply entirely in Thai", prompt)
        self.assertLess(len(prompt), 6000)


def _assistant_pack_call(call_id, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "function": {"name": "load_capability_pack", "arguments": arguments}}
        ],
    }


class CoachChatCapabilitiesTest(unittest.TestCase):
    """The typed chat has no session, so its tool surface is rebuilt from history."""

    def setUp(self):
        names = set(CORE_TOOL_NAMES)
        for pack_names in PACK_TOOL_NAMES.values():
            names.update(pack_names)
        self.tools = [_tool(name) for name in sorted(names)]

    def test_history_replays_loaded_packs_in_order_without_duplicates(self):
        history = [
            {"role": "user", "content": "how is my week going"},
            _assistant_pack_call("a", '{"pack":"Planning"}'),
            {"role": "tool", "tool_call_id": "a", "content": "{}"},
            _assistant_pack_call("b", '{"pack":"planning"}'),
            _assistant_pack_call("c", '{"pack":"discovery"}'),
        ]
        self.assertEqual(loaded_packs_from_history(history), ["planning", "discovery"])

    def test_history_ignores_junk(self):
        self.assertEqual(loaded_packs_from_history([]), [])
        self.assertEqual(loaded_packs_from_history(None), [])
        self.assertEqual(loaded_packs_from_history(["not a message"]), [])
        # Malformed arguments must not take the whole request down.
        self.assertEqual(loaded_packs_from_history([_assistant_pack_call("a", "not json")]), [])
        self.assertEqual(loaded_packs_from_history([_assistant_pack_call("a", '{"pack":""}')]), [])
        # A user message cannot smuggle a pack in — only the assistant loads tools.
        smuggled = dict(_assistant_pack_call("a", '{"pack":"marketplace"}'), role="user")
        self.assertEqual(loaded_packs_from_history([smuggled]), [])

    def test_first_round_is_core_plus_loader_only(self):
        tools, extra = coach_chat_capability_payload(self.tools, [])
        names = [tool["name"] for tool in tools]
        self.assertIn("load_capability_pack", names)
        self.assertNotIn("generate_plan", names)
        self.assertEqual(extra, "")

    def test_loaded_packs_add_their_tools_and_instructions_once(self):
        tools, extra = coach_chat_capability_payload(
            self.tools, ["planning", "planning", "unknown_pack", "discovery"]
        )
        names = [tool["name"] for tool in tools]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("generate_plan", names)
        self.assertIn("find_places", names)
        self.assertNotIn("book_offering", names)
        self.assertEqual(extra.count("PLANNING CAPABILITY"), 1)
        self.assertIn("DISCOVERY CAPABILITY", extra)

    def test_non_host_cannot_load_the_host_pack(self):
        tools, extra = coach_chat_capability_payload(self.tools, ["host"], is_host=False)
        self.assertNotIn("get_my_offerings", [tool["name"] for tool in tools])
        self.assertEqual(extra, "")


if __name__ == "__main__":
    unittest.main()
