"""Protection Zone (Claim 1e) — unit tests."""

from __future__ import annotations

import pytest

from tulbase.compose import (
    DEFAULT_MODE,
    PROTECTION_ZONE_N,
    compose_compresh_history,
    compose_compresh_system,
    resolve_n,
)
from tulbase.turn_box import TurnBox


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _box(turn: int, speaker: str, summary: str) -> TurnBox:
    return TurnBox(
        turn=turn,
        speaker=speaker,  # type: ignore[arg-type]
        session_id="sess-test",
        summary=summary,
    )


# ---------------------------------------------------------------------------
# resolve_n
# ---------------------------------------------------------------------------


def test_resolve_n_aggressive():
    assert resolve_n("aggressive") == 2


def test_resolve_n_balanced():
    assert resolve_n("balanced") == 4


def test_resolve_n_conservative():
    assert resolve_n("conservative") == 8


def test_resolve_n_integer_passthrough():
    assert resolve_n(6) == 6
    assert resolve_n(0) == 0


def test_resolve_n_negative_int_rejected():
    with pytest.raises(ValueError):
        resolve_n(-1)


def test_resolve_n_unknown_string_rejected():
    with pytest.raises(ValueError):
        resolve_n("turbo")


# ---------------------------------------------------------------------------
# compose_compresh_history — küçük geçmiş
# ---------------------------------------------------------------------------


def test_compose_empty_history():
    """upto_idx=0 → ne sıkıştırılan ne korunan mesaj."""
    out = compose_compresh_history([], [], upto_idx=0)
    assert out.compresh_md == ""
    assert out.raw_tail == []
    assert out.n_compressed == 0
    assert out.n_protected == 0


def test_compose_history_smaller_than_zone():
    """Geçmiş ≤ N → tamamı koruma bölgesinde, compress yok."""
    msgs = [_msg("user", "merhaba"), _msg("assistant", "selam")]
    boxes = [_box(0, "user", "kullanıcı selamlıyor"),
             _box(1, "assistant", "asistan karşılık veriyor")]
    out = compose_compresh_history(
        msgs, boxes, upto_idx=2, mode="balanced",
    )
    assert out.compresh_md == ""
    assert len(out.raw_tail) == 2
    assert out.raw_tail[0]["content"] == "merhaba"
    assert out.n_compressed == 0
    assert out.n_protected == 2


def test_compose_history_equal_to_zone():
    """Geçmiş tam N kadar → tamamı koruma bölgesinde."""
    msgs = [_msg("user", f"u{i}") for i in range(4)]
    boxes = [_box(i, "user", f"summary {i}") for i in range(4)]
    out = compose_compresh_history(
        msgs, boxes, upto_idx=4, mode="balanced",
    )
    assert out.compresh_md == ""
    assert len(out.raw_tail) == 4
    assert out.n_compressed == 0
    assert out.n_protected == 4


# ---------------------------------------------------------------------------
# compose_compresh_history — büyük geçmiş, split davranışı
# ---------------------------------------------------------------------------


def test_compose_history_larger_than_zone_balanced():
    """Geçmiş > N → eski [0..len-N] compress, son N raw."""
    msgs = [_msg("user", f"u{i}") if i % 2 == 0
            else _msg("assistant", f"a{i}") for i in range(10)]
    boxes = [_box(i, "user" if i % 2 == 0 else "assistant", f"sum{i}")
             for i in range(10)]
    out = compose_compresh_history(
        msgs, boxes, upto_idx=10, mode="balanced",
    )
    # 10 mesaj, N=4 → 6 compress, 4 raw
    assert out.n_compressed == 6
    assert out.n_protected == 4
    assert len(out.raw_tail) == 4
    # Son 4 mesaj raw tail'de olmalı
    assert out.raw_tail[0]["content"] == "u6"
    assert out.raw_tail[3]["content"] == "a9"
    # compresh_md sum0..sum5 başlıklarını içermeli
    assert "sum0" in out.compresh_md or "[T0" in out.compresh_md
    assert "[T5" in out.compresh_md
    # Koruma bölgesindeki turn'lerin markdown'ı OLMAMALI
    assert "[T6" not in out.compresh_md
    assert "[T9" not in out.compresh_md


def test_compose_history_aggressive_mode():
    """Agresif mod N=2 → daha az koruma, daha çok sıkıştırma."""
    msgs = [_msg("user", f"u{i}") for i in range(6)]
    boxes = [_box(i, "user", f"sum{i}") for i in range(6)]
    out = compose_compresh_history(
        msgs, boxes, upto_idx=6, mode="aggressive",
    )
    assert out.n_compressed == 4
    assert out.n_protected == 2
    assert out.raw_tail[0]["content"] == "u4"
    assert out.raw_tail[1]["content"] == "u5"
    assert out.mode == "aggressive"


def test_compose_history_conservative_mode():
    """Muhafazakâr mod N=8 → daha fazla koruma."""
    msgs = [_msg("user", f"u{i}") for i in range(12)]
    boxes = [_box(i, "user", f"sum{i}") for i in range(12)]
    out = compose_compresh_history(
        msgs, boxes, upto_idx=12, mode="conservative",
    )
    assert out.n_compressed == 4
    assert out.n_protected == 8
    assert len(out.raw_tail) == 8
    assert out.mode == "conservative"


def test_compose_history_integer_override():
    """N integer doğrudan geçilirse mod tablosu yerine kullanılır."""
    msgs = [_msg("user", f"u{i}") for i in range(10)]
    boxes = [_box(i, "user", f"sum{i}") for i in range(10)]
    out = compose_compresh_history(
        msgs, boxes, upto_idx=10, mode=3,
    )
    assert out.n_compressed == 7
    assert out.n_protected == 3


def test_compose_history_default_mode_balanced():
    """Mod verilmezse default balanced (N=4)."""
    msgs = [_msg("user", f"u{i}") for i in range(10)]
    boxes = [_box(i, "user", f"sum{i}") for i in range(10)]
    out = compose_compresh_history(msgs, boxes, upto_idx=10)
    assert DEFAULT_MODE == "balanced"
    assert out.n_compressed == 6
    assert out.n_protected == 4


def test_compose_history_subset_via_upto_idx():
    """upto_idx tüm mesaj listesinden küçük olabilir."""
    msgs = [_msg("user", f"u{i}") for i in range(10)]
    boxes = [_box(i, "user", f"sum{i}") for i in range(10)]
    out = compose_compresh_history(msgs, boxes, upto_idx=6, mode="balanced")
    # Sadece ilk 6 mesaj — 2 compress + 4 raw
    assert out.n_compressed == 2
    assert out.n_protected == 4
    assert out.raw_tail[0]["content"] == "u2"
    assert out.raw_tail[3]["content"] == "u5"


# ---------------------------------------------------------------------------
# Hata yolları
# ---------------------------------------------------------------------------


def test_compose_upto_idx_negative_rejected():
    with pytest.raises(ValueError):
        compose_compresh_history([], [], upto_idx=-1)


def test_compose_upto_idx_too_large_rejected():
    msgs = [_msg("user", "x")]
    with pytest.raises(ValueError):
        compose_compresh_history(msgs, [], upto_idx=5)


# ---------------------------------------------------------------------------
# compose_compresh_system
# ---------------------------------------------------------------------------


def test_compose_system_with_compressed_history():
    msgs = [_msg("user", f"u{i}") for i in range(10)]
    boxes = [_box(i, "user", f"sum{i}") for i in range(10)]
    composed = compose_compresh_history(msgs, boxes, upto_idx=10)
    sys_prompt = compose_compresh_system(
        "You are an assistant.", composed,
    )
    assert "You are an assistant." in sys_prompt
    assert "compressed memory" in sys_prompt.lower()
    assert "[T0" in sys_prompt
    # son N koruma bölgesindeki turn'ler system'de OLMAMALI
    assert "[T9" not in sys_prompt


def test_compose_system_no_history():
    """Geçmiş tamamen koruma bölgesinde → system'de history bloğu yok."""
    msgs = [_msg("user", f"u{i}") for i in range(2)]
    boxes = [_box(i, "user", f"sum{i}") for i in range(2)]
    composed = compose_compresh_history(msgs, boxes, upto_idx=2)
    sys_prompt = compose_compresh_system(
        "Base system.", composed,
    )
    assert "Base system." in sys_prompt
    # compresh_md boş — history bloğu yazılmamalı
    assert "compressed memory" not in sys_prompt.lower()


def test_compose_system_honesty_fragment_appended():
    composed = compose_compresh_history([], [], upto_idx=0)
    sys_prompt = compose_compresh_system(
        "Base.", composed, honesty_fragment="HONESTY NOTE",
    )
    assert sys_prompt.endswith("HONESTY NOTE")


def test_compose_system_empty_history_note():
    composed = compose_compresh_history([], [], upto_idx=0)
    sys_prompt = compose_compresh_system(
        "Base.", composed, empty_history_note="No prior chat.",
    )
    assert "No prior chat." in sys_prompt


# ---------------------------------------------------------------------------
# Mode constants (patent compliance)
# ---------------------------------------------------------------------------


def test_protection_zone_constants_match_patent_claim_1e():
    """Claim 1(e): aggressive=2, balanced=4, conservative=8."""
    assert PROTECTION_ZONE_N["aggressive"] == 2
    assert PROTECTION_ZONE_N["balanced"] == 4
    assert PROTECTION_ZONE_N["conservative"] == 8


def test_default_mode_is_balanced():
    """Default mod patent'te explicitly belirtilmez ama dengeli mantıklı."""
    assert DEFAULT_MODE == "balanced"
