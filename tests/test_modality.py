"""Tests for the modality classifier (modality.py)."""

from __future__ import annotations

import unittest

from tulbase.modality import (
    PATTERNS,
    Segment,
    classify,
    resolve_overlaps,
)


class TestPatternCoverage(unittest.TestCase):
    """Each pattern must match its archetype example."""

    def test_code_block(self):
        text = "Here is code:\n```python\nprint('hi')\n```\nDone."
        segs = classify(text)
        modalities = {s.modality for s in segs}
        self.assertIn("code_block", modalities)

    def test_inline_code(self):
        text = "Run `pytest -x` to stop on first fail."
        segs = classify(text)
        self.assertTrue(any(s.modality == "inline_code" for s in segs))

    def test_url(self):
        text = "See https://compre.sh/docs for details."
        segs = classify(text)
        self.assertTrue(any(s.modality == "url" for s in segs))

    def test_stack_trace(self):
        text = (
            "Got error:\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 3, in <module>\n'
            "    1 / 0\n"
            "ZeroDivisionError: division by zero\n"
            "\n"
            "Next paragraph."
        )
        segs = classify(text)
        self.assertTrue(any(s.modality == "stack_trace" for s in segs))

    def test_terminal_output(self):
        text = "Look at this:\n❯ ls -la\ntotal 4\nfile.txt\n"
        segs = classify(text)
        self.assertTrue(any(s.modality == "terminal_output" for s in segs))

    def test_json_dump_long_enough(self):
        # Build a JSON object > 500 chars so the pattern fires.
        big = "{\n" + ",\n".join(f'  "k{i}": "v{i}"' for i in range(80)) + "\n}"
        self.assertGreater(len(big), 500)
        text = "Response payload:\n" + big + "\nDone."
        segs = classify(text)
        self.assertTrue(any(s.modality == "json_dump" for s in segs))

    def test_json_too_short_no_match(self):
        text = 'Body: {\n  "k": 1\n}\nEnd.'
        segs = classify(text)
        self.assertFalse(any(s.modality == "json_dump" for s in segs))


class TestEmpty(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(classify(""), [])

    def test_pure_dialog(self):
        text = "Selam, nasılsın? Bugün hava güzel."
        segs = classify(text)
        # No code/terminal/json — only an inline-code-less question marker.
        # We allow zero segments here; question-stem detection lives in
        # summarizer, not modality classifier.
        self.assertEqual([s for s in segs if s.modality in
                          {"code_block", "terminal_output", "json_dump",
                           "stack_trace"}], [])


class TestOverlapResolution(unittest.TestCase):
    def test_inline_code_inside_code_block_dropped(self):
        text = "```python\nx = `inner`\n```"
        segs = classify(text)
        modalities = [s.modality for s in segs]
        # code_block kept, inner inline_code dropped (strictly contained).
        self.assertIn("code_block", modalities)
        self.assertNotIn("inline_code", modalities)

    def test_url_inside_code_block_dropped(self):
        text = "```\nopen https://example.com page\n```"
        segs = classify(text)
        self.assertEqual([s.modality for s in segs], ["code_block"])

    def test_two_disjoint_segments_both_kept(self):
        text = "Code:\n```py\nfoo()\n```\nMore.\n```py\nbar()\n```"
        segs = classify(text)
        self.assertEqual(
            [s.modality for s in segs], ["code_block", "code_block"]
        )

    def test_specificity_wins_on_partial_overlap(self):
        # Manually craft overlapping segments to test resolve_overlaps.
        a = Segment(0, 50, "url", 0.95, text="x" * 50)
        b = Segment(20, 80, "code_block", 0.95, text="y" * 60)
        out = resolve_overlaps([a, b])
        # code_block has higher specificity → kept.
        self.assertEqual([s.modality for s in out], ["code_block"])


class TestPatternsAreDefined(unittest.TestCase):
    def test_six_patterns_present(self):
        for key in (
            "code_block",
            "terminal_output",
            "json_dump",
            "stack_trace",
            "url",
            "inline_code",
        ):
            self.assertIn(key, PATTERNS, f"missing pattern: {key}")


class TestShellPromptCoverage(unittest.TestCase):
    """Production shell prompts ($, #, >) must match — not just ❯.

    Regression test for bug fix in 2026-05-09 review (BLOCKER #2).
    """

    def test_dollar_prompt(self):
        text = "Look:\n$ docker ps\nCONTAINER ID\nabc123\n"
        segs = classify(text)
        self.assertTrue(
            any(s.modality == "terminal_output" for s in segs),
            f"$ prompt not matched: {[s.modality for s in segs]}",
        )

    def test_hash_prompt_root(self):
        text = "As root:\n# pip install duckdb\nSuccessfully installed\n"
        segs = classify(text)
        self.assertTrue(
            any(s.modality == "terminal_output" for s in segs),
            f"# prompt not matched: {[s.modality for s in segs]}",
        )

    def test_gt_prompt_powershell(self):
        text = "Run:\n> npm test\nPASS test/foo.spec.js\n"
        segs = classify(text)
        self.assertTrue(
            any(s.modality == "terminal_output" for s in segs),
            f"> prompt not matched: {[s.modality for s in segs]}",
        )

    def test_warp_prompt_still_works(self):
        text = "❯ ls\ntotal 4\nfile.txt\n"
        segs = classify(text)
        self.assertTrue(
            any(s.modality == "terminal_output" for s in segs)
        )

    def test_mixed_prompts_in_one_turn(self):
        text = (
            "First:\n❯ docker ps\nNAME\napi\n\n"
            "Then:\n$ ls -la\ntotal 4\n\n"
            "And code:\n```python\nx = 1\n```"
        )
        segs = classify(text)
        mods = [s.modality for s in segs]
        self.assertEqual(
            mods.count("terminal_output"),
            2,
            f"expected 2 terminal blocks, got {mods}",
        )
        self.assertIn("code_block", mods)


if __name__ == "__main__":
    unittest.main()
