"""End-to-end test for Pipeline (pipeline.py)."""

from __future__ import annotations

import os
import tempfile
import unittest

try:
    import duckdb  # noqa: F401
    DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    DUCKDB_AVAILABLE = False

from tulbase.cold_storage import ColdStorage
from tulbase.compression_log import CompressionLog
from tulbase.pipeline import Pipeline
from tulbase.retrieval import Retriever
from tulbase.summarizer import Tier1Summarizer
from tulbase.turn_box import render_markdown

SAMPLE_TURN = """\
Phase 2.1 v3 deploy script hazırlandı. Şu şekilde:

```bash
rsync -avz --delete /local/ user@host:/opt/compresh-api/
ssh user@host 'docker compose restart'
```

Acaba v3 testi 105 QA için yeterli mi? Bu konu hâlâ açık.

Terminal output:
❯ docker compose ps
NAME       STATUS
api        Up 5 minutes
db         Up 5 minutes
"""


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        # Temp DuckDB — DuckDB ≥1.5 rejects empty files, unlink first.
        self.db_tmp = tempfile.NamedTemporaryFile(
            suffix=".duckdb", delete=False
        )
        self.db_tmp.close()
        self.db_path = self.db_tmp.name
        os.unlink(self.db_path)

        # Temp cold dir
        self.cold_dir = tempfile.mkdtemp(prefix="compresh_cold_")

        self.log = CompressionLog(self.db_path)
        self.log.ensure_schema()
        self.cold = ColdStorage(self.cold_dir)
        self.pipeline = Pipeline(self.log, self.cold, Tier1Summarizer())

    def tearDown(self):
        self.log.close()
        os.unlink(self.db_path)
        # Best-effort cold dir cleanup.
        for dirpath, _, files in os.walk(self.cold_dir, topdown=False):
            for f in files:
                os.unlink(os.path.join(dirpath, f))
            os.rmdir(dirpath)

    def test_run_compresses_code_and_terminal(self):
        result = self.pipeline.run(
            SAMPLE_TURN,
            session_id="sess-test",
            turn_idx=1,
            speaker="assistant",
        )

        # We expect at least one code_block and one terminal_output entry.
        modalities = sorted(e.modality for e in result.entries)
        self.assertIn("code", modalities)
        self.assertIn("terminal_output", modalities)

        # The turn box must reference the same number of entries.
        self.assertEqual(len(result.turn_box.compressed_refs), len(result.entries))

        # Markdown must contain the [TN (Assistant)] header and Compressed line.
        md = result.markdown
        self.assertIn("[T1 (Assistant)]", md)
        self.assertIn("Compressed:", md)

        # Saving — at the entry level (cold-stored content vs short marker).
        # Whole-turn `size_compressed` may exceed `size_orig` for tiny turns
        # where the markdown overhead (Summary/Compressed/Opens lines) is
        # bigger than the elided payload — that's a known property and
        # only matters for *large* turns. The honest invariant is that
        # each individual elided segment is smaller as a marker than as
        # raw bytes:
        for entry in result.entries:
            self.assertLess(
                entry.size_compressed,
                entry.size_orig,
                f"entry {entry.id}: marker {entry.size_compressed} ≥ orig {entry.size_orig}",
            )

    def test_open_question_extracted(self):
        result = self.pipeline.run(
            SAMPLE_TURN,
            session_id="sess-test",
            turn_idx=2,
            speaker="user",
        )
        # "Acaba v3 testi 105 QA için yeterli mi?" is a question — must
        # surface in opens.
        self.assertTrue(
            any("v3" in q or "yeterli" in q for q in result.turn_box.opens),
            f"opens={result.turn_box.opens!r}",
        )

    def test_retrieval_round_trip(self):
        result = self.pipeline.run(
            SAMPLE_TURN,
            session_id="sess-rt",
            turn_idx=3,
            speaker="assistant",
        )
        self.assertGreater(len(result.entries), 0)

        retriever = Retriever(self.log, self.cold)
        first_id = result.entries[0].id
        rr = retriever.fetch(first_id)
        self.assertTrue(rr.ok, f"error={rr.error}")
        self.assertIsNotNone(rr.content)

    def test_pii_blocked_retrieval(self):
        result = self.pipeline.run(
            SAMPLE_TURN,
            session_id="sess-pii",
            turn_idx=4,
            speaker="assistant",
        )
        first_id = result.entries[0].id
        self.log.mark_pii_filtered(first_id)

        retriever = Retriever(self.log, self.cold)
        rr = retriever.fetch(first_id)
        self.assertFalse(rr.ok)
        self.assertIn("retrievable", rr.error or "")

    def test_dedup_same_content_one_cold_object(self):
        # Run the same turn twice — cold storage should dedup by hash.
        r1 = self.pipeline.run(
            SAMPLE_TURN,
            session_id="sess-dedup",
            turn_idx=1,
            speaker="user",
        )
        r2 = self.pipeline.run(
            SAMPLE_TURN,
            session_id="sess-dedup",
            turn_idx=2,
            speaker="user",
        )
        # Same hash → same cold object reused (dedup).
        hashes_1 = {e.hash for e in r1.entries}
        hashes_2 = {e.hash for e in r2.entries}
        self.assertEqual(hashes_1, hashes_2)
        # But the compression_log rows are distinct (per-turn).
        ids_1 = {e.id for e in r1.entries}
        ids_2 = {e.id for e in r2.entries}
        self.assertFalse(ids_1 & ids_2)

    def test_render_markdown_round_trip(self):
        result = self.pipeline.run(
            SAMPLE_TURN,
            session_id="sess-md",
            turn_idx=7,
            speaker="assistant",
        )
        md_again = render_markdown(result.turn_box)
        self.assertEqual(result.markdown, md_again)


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class TestPipelineEdgeCases(unittest.TestCase):
    def setUp(self):
        self.db_tmp = tempfile.NamedTemporaryFile(
            suffix=".duckdb", delete=False
        )
        self.db_tmp.close()
        os.unlink(self.db_tmp.name)
        self.cold_dir = tempfile.mkdtemp(prefix="compresh_cold_")
        self.log = CompressionLog(self.db_tmp.name)
        self.log.ensure_schema()
        self.cold = ColdStorage(self.cold_dir)
        self.pipeline = Pipeline(self.log, self.cold)

    def tearDown(self):
        self.log.close()
        os.unlink(self.db_tmp.name)
        for dirpath, _, files in os.walk(self.cold_dir, topdown=False):
            for f in files:
                os.unlink(os.path.join(dirpath, f))
            os.rmdir(dirpath)

    def test_empty_turn(self):
        result = self.pipeline.run(
            "",
            session_id="sess-empty",
            turn_idx=0,
            speaker="user",
        )
        self.assertEqual(len(result.entries), 0)
        self.assertIn("[T0 (User)]", result.markdown)

    def test_pure_dialog_no_compression(self):
        result = self.pipeline.run(
            "Selam, bugün havaya çıktın mı?",
            session_id="sess-dialog",
            turn_idx=0,
            speaker="user",
        )
        self.assertEqual(len(result.entries), 0)
        self.assertEqual(len(result.turn_box.compressed_refs), 0)

    def test_invalid_speaker_raises(self):
        with self.assertRaises(ValueError):
            self.pipeline.run(
                "x", session_id="s", turn_idx=0, speaker="alien"
            )

    def test_negative_turn_raises(self):
        with self.assertRaises(ValueError):
            self.pipeline.run(
                "x", session_id="s", turn_idx=-1, speaker="user"
            )


if __name__ == "__main__":
    unittest.main()
