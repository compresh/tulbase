"""Security regression tests — added 2026-05-09 review.

Covers:
  - Path-traversal protection in cold_storage (BLOCKER #1)
  - Question-detection word-boundary in summarizer (BLOCKER #4)
"""

from __future__ import annotations

import tempfile
import unittest

from tulbase.cold_storage import ColdStorage
from tulbase.summarizer import Tier1Summarizer, _is_question


class TestColdStoragePathTraversal(unittest.TestCase):
    """session_id must not allow escaping the cold root."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="compresh_cold_sec_")
        self.cs = ColdStorage(self.tmp)

    def test_dotdot_rejected(self):
        with self.assertRaises(ValueError):
            self.cs.save("../../../etc/passwd", "x")

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            self.cs.save("/absolute/path", "x")

    def test_dotdot_alone_rejected(self):
        with self.assertRaises(ValueError):
            self.cs.save("..", "x")

    def test_null_byte_rejected(self):
        with self.assertRaises(ValueError):
            self.cs.save("session\x00inj", "x")

    def test_shell_meta_rejected(self):
        with self.assertRaises(ValueError):
            self.cs.save("session;rm -rf", "x")

    def test_valid_session_id_works(self):
        obj = self.cs.save("v1:cowork:abdullah", "test content")
        self.assertTrue(str(obj.absolute_path).startswith(str(self.cs.root)))


class TestQuestionDetection(unittest.TestCase):
    """Question detection must not produce false positives on prose
    that happens to contain English question words (BLOCKER #4)."""

    def test_where_in_tr_prose_not_question(self):
        s = "Cümle bazlı 5W1H slot doluluk düşüktü, where %0.9 idi."
        self.assertFalse(_is_question(s), f"FP: {s!r}")

    def test_where_mid_sentence_en_not_question(self):
        s = "The location, where the event happened, was unclear."
        self.assertFalse(_is_question(s), f"FP: {s!r}")

    def test_question_mark_is_question(self):
        self.assertTrue(_is_question("Ne dersin?"))
        self.assertTrue(_is_question("Is that right?"))

    def test_en_start_of_sentence_is_question(self):
        self.assertTrue(_is_question("How does this work"))
        self.assertTrue(_is_question("Where is the data"))

    def test_tr_particle_is_question(self):
        self.assertTrue(_is_question("Geliyor musun bugün"))
        self.assertTrue(_is_question("Bitti mi"))

    def test_summarizer_filters_fp(self):
        sumz = Tier1Summarizer()
        text = (
            "Phase 2.2 ortaya çıktı. "
            "Cümle bazlı slot doluluk düşüktü, where %0.9 idi. "
            "Ne dersin? Bench testlerine ne zaman başlayacağız?"
        )
        r = sumz.run(text)
        # FP cümle olmamalı
        self.assertFalse(
            any("doluluk düşüktü" in o for o in r.opens),
            f"false-positive: {r.opens}",
        )
        # Gerçek sorular yakalanmalı
        self.assertTrue(any("dersin" in o for o in r.opens))
        self.assertTrue(any("zaman" in o for o in r.opens))


class TestPipelineShortLabel(unittest.TestCase):
    """_SHORT_LABEL must contain the post-rename modality keys
    (BLOCKER #3 — README example used 'code block' string)."""

    def test_short_label_has_code_key(self):
        from tulbase.pipeline import _SHORT_LABEL
        self.assertEqual(_SHORT_LABEL.get("code"), "code block")
        self.assertEqual(_SHORT_LABEL.get("terminal_output"), "terminal output")


if __name__ == "__main__":
    unittest.main()
