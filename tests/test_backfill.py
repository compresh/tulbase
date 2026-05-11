"""Tests for backfill.py — retrospective batch compression."""

from __future__ import annotations

import os
import tempfile
import unittest

try:
    import duckdb  # noqa: F401
    DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    DUCKDB_AVAILABLE = False

from tulbase.backfill import Backfiller, _normalize_message, backfill_messages
from tulbase.cold_storage import ColdStorage
from tulbase.compression_log import CompressionLog


# ---------------------------------------------------------------------------
# Pure-function tests — no DB needed
# ---------------------------------------------------------------------------


class TestNormalizeMessage(unittest.TestCase):
    """Source-attribution shim — pipeline_speaker, text, and provenance."""

    def test_plain_string_user(self):
        speaker, text, prov = _normalize_message(
            {"role": "user", "content": "hello"}
        )
        self.assertEqual(speaker, "user")
        self.assertEqual(text, "hello")
        self.assertEqual(prov.channel, "user")
        self.assertEqual(prov.trust_level, "user_input")

    def test_plain_string_assistant(self):
        speaker, text, prov = _normalize_message(
            {"role": "assistant", "content": "hi back"}
        )
        self.assertEqual(speaker, "assistant")
        self.assertEqual(text, "hi back")
        self.assertEqual(prov.channel, "assistant")
        self.assertEqual(prov.trust_level, "model_generated")

    def test_text_block_list(self):
        speaker, text, prov = _normalize_message({
            "role": "assistant",
            "content": [{"type": "text", "text": "answer here"}],
        })
        self.assertEqual(speaker, "assistant")
        self.assertEqual(text, "answer here")
        self.assertEqual(prov.channel, "assistant")

    def test_tool_use_block_keeps_assistant_channel(self):
        speaker, text, prov = _normalize_message({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will check."},
                {"type": "tool_use", "name": "fetch_compressed",
                 "input": {"id": "compr-T5-code-000"}},
            ],
        })
        self.assertEqual(speaker, "assistant")
        self.assertEqual(prov.channel, "assistant")
        self.assertEqual(prov.source_name, "fetch_compressed")
        self.assertIn("I will check.", text)
        self.assertIn("fetch_compressed", text)

    def test_tool_result_separates_channel_from_pipeline_speaker(self):
        # Wire format: tool_result rides in a user-role message — pipeline
        # speaker stays "user" but the provenance channel must be "tool".
        speaker, text, prov = _normalize_message({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_123",
                 "content": "the original code was: def foo(): pass"},
            ],
        })
        self.assertEqual(speaker, "user", "pipeline-facing speaker")
        self.assertEqual(prov.channel, "tool", "source-aware channel")
        self.assertIn("def foo()", text)
        self.assertEqual(prov.trust_level, "tool_validated")

    def test_explicit_channel_override(self):
        # Caller-supplied channel hint always wins.
        speaker, text, prov = _normalize_message({
            "role": "user",
            "content": "Bitcoin closed at 73K",
            "channel": "external",
            "source_name": "coinbase.com",
        })
        self.assertEqual(speaker, "user")
        self.assertEqual(prov.channel, "external")
        self.assertEqual(prov.source_name, "coinbase.com")
        self.assertEqual(prov.trust_level, "external_fetched")

    def test_scheduled_channel(self):
        speaker, text, prov = _normalize_message({
            "role": "user",
            "content": "Morning brief: 3 PRs awaiting review",
            "channel": "scheduled",
            "source_name": "morning-brief",
        })
        self.assertEqual(speaker, "user")
        self.assertEqual(prov.channel, "scheduled")
        self.assertEqual(prov.source_name, "morning-brief")

    def test_none_content(self):
        speaker, text, prov = _normalize_message(
            {"role": "user", "content": None}
        )
        self.assertEqual(speaker, "user")
        self.assertEqual(text, "")
        self.assertEqual(prov.channel, "user")

    def test_unknown_role_defaults_user(self):
        speaker, text, prov = _normalize_message({"content": "x"})
        self.assertEqual(speaker, "user")
        self.assertEqual(text, "x")
        self.assertEqual(prov.channel, "user")

    def test_parent_turn_propagates(self):
        _, _, prov = _normalize_message(
            {"role": "user", "content": "x"}, turn_idx=42,
        )
        self.assertEqual(prov.parent_turn, 42)


# ---------------------------------------------------------------------------
# Full backfill — needs DuckDB
# ---------------------------------------------------------------------------


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class TestBackfiller(unittest.TestCase):
    def setUp(self):
        self.db_tmp = tempfile.NamedTemporaryFile(
            suffix=".duckdb", delete=False
        )
        self.db_tmp.close()
        self.db_path = self.db_tmp.name
        os.unlink(self.db_path)
        self.cold_dir = tempfile.mkdtemp(prefix="compresh_backfill_test_")
        self.log = CompressionLog(self.db_path)
        self.log.ensure_schema()
        self.cold = ColdStorage(self.cold_dir)
        self.bf = Backfiller(log=self.log, cold=self.cold)

    def tearDown(self):
        try:
            self.log.close()
        except Exception:
            pass
        for p in [self.db_path, self.db_path + ".wal"]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def test_empty_messages_produces_empty_result(self):
        result = self.bf.run_batch([], session_id="empty-test")
        self.assertEqual(len(result.boxes), 0)
        self.assertEqual(result.total_size_orig, 0)
        self.assertEqual(result.saving_ratio, 0.0)

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello world"}]
        result = self.bf.run_batch(msgs, session_id="single-test")
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0].turn, 0)
        self.assertEqual(result.boxes[0].speaker, "user")

    def test_multi_message_preserves_order(self):
        msgs = [
            {"role": "user", "content": "first user msg"},
            {"role": "assistant", "content": "first assistant msg"},
            {"role": "user", "content": "second user msg"},
        ]
        result = self.bf.run_batch(msgs, session_id="multi-test")
        self.assertEqual(len(result.boxes), 3)
        self.assertEqual([b.turn for b in result.boxes], [0, 1, 2])
        self.assertEqual(
            [b.speaker for b in result.boxes],
            ["user", "assistant", "user"],
        )

    def test_session_id_required(self):
        with self.assertRaises(ValueError):
            self.bf.run_batch(
                [{"role": "user", "content": "x"}], session_id=""
            )

    def test_long_distance_resolves_via_accumulator(self):
        # Turn 0 opens a question with a named entity.
        # Turns 1,2,3 don't mention it.
        # Turn 4 answers using the same entity name.
        # Single-pass realtime would lose this — only previous-turn
        # opens is fed. Accumulator should preserve the open across
        # the gap.
        msgs = [
            {"role": "user", "content": "What is the status of Phase22?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Still checking."},
            {"role": "assistant", "content": "Phase22 is fully deployed."},
        ]
        result = self.bf.run_batch(msgs, session_id="lr-test")
        self.assertEqual(len(result.boxes), 5)
        # Turn 0 should produce an open
        opens_T0 = result.boxes[0].opens
        self.assertGreater(len(opens_T0), 0,
                           "Turn 0 should detect the question")
        # Turn 4 should resolve it (entity 'Phase22' matches across the gap)
        resolves_T4 = result.boxes[4].resolves
        self.assertGreater(len(resolves_T4), 0,
                           "Turn 4 should resolve the long-distance open "
                           "thanks to the accumulator")

    def test_compressed_refs_for_code_block(self):
        msgs = [
            {"role": "user", "content": "show me the code"},
            {"role": "assistant", "content":
                "Here it is:\n```python\ndef foo():\n    return 42\n```\n"},
        ]
        result = self.bf.run_batch(msgs, session_id="refs-test")
        # The assistant turn should have at least one compressed_ref
        self.assertEqual(len(result.boxes), 2)
        self.assertGreater(
            len(result.boxes[1].compressed_refs), 0,
            "Code block should have been elided into a compressed_ref",
        )

    def test_tool_result_does_not_crash(self):
        msgs = [
            {"role": "user", "content": "what was the output?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Fetching..."},
                {"type": "tool_use", "name": "fetch_compressed",
                 "input": {"id": "compr-T5-code-000"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": "def foo(): return 42"},
            ]},
            {"role": "assistant", "content": "The function returns 42."},
        ]
        result = self.bf.run_batch(msgs, session_id="tool-test")
        self.assertEqual(len(result.boxes), 4)
        self.assertEqual(result.skipped_turns, [])

    def test_functional_helper(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = backfill_messages(
            msgs, session_id="fn-test",
            log=self.log, cold=self.cold,
        )
        self.assertEqual(len(result.boxes), 1)

    def test_provenance_propagates_to_turnbox(self):
        """Each TurnBox should carry the inferred provenance."""
        msgs = [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
            {
                "role": "user",
                "content": "external fact",
                "channel": "external",
                "source_name": "wikipedia.org",
            },
        ]
        result = self.bf.run_batch(msgs, session_id="prov-prop-test")
        self.assertEqual(len(result.boxes), 3)
        # Box 0 — user keyboard
        self.assertIsNotNone(result.boxes[0].provenance)
        self.assertEqual(result.boxes[0].provenance.channel, "user")
        # Box 1 — assistant generated
        self.assertEqual(result.boxes[1].provenance.channel, "assistant")
        # Box 2 — external fetch, pipeline speaker still "user" but
        # provenance separates the channel
        self.assertEqual(result.boxes[2].speaker, "user")
        self.assertEqual(result.boxes[2].provenance.channel, "external")
        self.assertEqual(result.boxes[2].provenance.source_name, "wikipedia.org")

    def test_provenance_persists_in_compression_log(self):
        """When the pipeline emits a CompressionEntry, the provenance
        must round-trip through the DB metadata column."""
        msgs = [
            {"role": "user", "content": "show the code"},
            {
                "role": "assistant",
                "content":
                    "Here:\n```python\ndef foo():\n    return 1\n```",
            },
        ]
        result = self.bf.run_batch(msgs, session_id="prov-db-test")
        # Assistant turn should have produced an entry
        self.assertEqual(len(result.results), 2)
        entries = result.results[1].entries
        self.assertGreater(len(entries), 0)
        # Re-read via CompressionLog.get to ensure persistence
        roundtripped = self.log.get(entries[0].id)
        self.assertIsNotNone(roundtripped)
        self.assertIsNotNone(roundtripped.provenance)
        self.assertEqual(roundtripped.provenance.channel, "assistant")


if __name__ == "__main__":
    unittest.main()
