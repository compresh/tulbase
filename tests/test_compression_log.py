"""Tests for compression_log.py (DuckDB schema + CRUD)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone

try:
    import duckdb  # noqa: F401
    DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    DUCKDB_AVAILABLE = False

from tulbase.compression_log import (
    CompressionEntry,
    CompressionLog,
)


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class TestSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        os.unlink(self.path)  # DuckDB ≥1.5 rejects empty files
        self.log = CompressionLog(self.path)
        self.log.ensure_schema()

    def tearDown(self):
        self.log.close()
        os.unlink(self.path)

    def test_idempotent_schema(self):
        # Calling twice must not raise.
        self.log.ensure_schema()
        self.log.ensure_schema()


@unittest.skipUnless(DUCKDB_AVAILABLE, "duckdb not installed")
class TestCRUD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
        self.tmp.close()
        self.path = self.tmp.name
        os.unlink(self.path)  # DuckDB ≥1.5 rejects empty files
        self.log = CompressionLog(self.path)
        self.log.ensure_schema()

    def tearDown(self):
        self.log.close()
        os.unlink(self.path)

    def _make_entry(self, **kw) -> CompressionEntry:
        defaults = dict(
            id="compr-sess1-T1-code-000",
            session_id="sess1",
            turn_idx=1,
            modality="code",
            summary="code block: print('hi')",
            reason="code modality removal",
            hash="a" * 64,
            size_orig=200,
            size_compressed=30,
            retrievable=True,
            pii_filtered=False,
            created_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
            cold_path="sess1/aa/aaaa.bin",
            metadata={"language": "python"},
        )
        defaults.update(kw)
        return CompressionEntry(**defaults)

    def test_save_and_get(self):
        e = self._make_entry()
        self.log.save(e)
        got = self.log.get(e.id)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.id, e.id)
        self.assertEqual(got.session_id, "sess1")
        self.assertEqual(got.modality, "code")
        self.assertEqual(got.metadata, {"language": "python"})

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.log.get("nope"))

    def test_list_by_session_filter_turn_range(self):
        for i in range(5):
            self.log.save(
                self._make_entry(
                    id=f"compr-sess1-T{i}-code-000",
                    turn_idx=i,
                    hash=("b" * 63) + str(i),
                )
            )
        rows = self.log.list_by_session("sess1", turn_min=1, turn_max=3)
        self.assertEqual([r.turn_idx for r in rows], [1, 2, 3])

    def test_list_by_session_filter_modality(self):
        self.log.save(self._make_entry(id="x1", modality="code", hash="c" * 64))
        self.log.save(
            self._make_entry(
                id="x2", modality="terminal_output", hash="d" * 64
            )
        )
        rows = self.log.list_by_session("sess1", modality="terminal_output")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, "x2")

    def test_find_by_hash_dedup(self):
        h = "e" * 64
        self.log.save(self._make_entry(id="x1", hash=h, turn_idx=1))
        self.log.save(self._make_entry(id="x2", hash=h, turn_idx=2))
        rows = self.log.find_by_hash(h)
        self.assertEqual({r.id for r in rows}, {"x1", "x2"})

    def test_mark_pii_filtered_blocks_retrieval(self):
        e = self._make_entry()
        self.log.save(e)
        self.log.mark_pii_filtered(e.id)
        got = self.log.get(e.id)
        assert got is not None
        self.assertTrue(got.pii_filtered)
        self.assertFalse(got.retrievable)

    def test_delete(self):
        e = self._make_entry()
        self.log.save(e)
        self.log.delete(e.id)
        self.assertIsNone(self.log.get(e.id))

    def test_duplicate_id_raises(self):
        e = self._make_entry()
        self.log.save(e)
        with self.assertRaises(Exception):
            self.log.save(e)

    def test_parameter_binding_safe(self):
        # This string would break naive string-formatted SQL.
        nasty = "x'); DROP TABLE compression_log; --"
        e = self._make_entry(id=nasty, hash="f" * 64)
        self.log.save(e)
        got = self.log.get(nasty)
        self.assertIsNotNone(got)


if __name__ == "__main__":
    unittest.main()
