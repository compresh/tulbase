"""Protection Zone — patent Talep 1(e) implementasyonu.

Konuşmanın son N mesajı bir koruma bölgesi olarak işaretlenir ve
sıkıştırmadan muaf tutulur. Geriye kalan eski mesajlar TurnBox
markdown formuna çevrilir. N değeri sıkıştırma moduna göre değişir:

    aggressive    → N = 2
    balanced      → N = 4   (default)
    conservative  → N = 8

İlgili patent kaynağı:
    compresh-ltd/legal/patents/provisional-uk/claims-draft-v3.md
    Talep 1, adım (e) Protection Zone
    Talep 5, mode konfigürasyonu

Tasarım:
  - Stateless helper — pipeline veya bench runner çağırır
  - Mesaj sayısı ≤ N ise koruma bölgesi tüm geçmişi kapsar
    (compress edilecek hiçbir mesaj yok, compresh_md boş)
  - Mesaj sayısı > N ise eski [0..len-N] mesajlar TurnBox'a
    compress edilir, son N raw kalır
  - Helper, hem markdown'ı hem raw tail'i döner — caller bunları
    provider mesaj formatına gömer (OpenAI: messages list,
    Anthropic: system + messages)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

from .turn_box import TurnBox, render_markdown_many

ProtectionMode = Literal["aggressive", "balanced", "conservative"]

# Patent Talep 5: mod-N eşlemesi.
PROTECTION_ZONE_N: dict[str, int] = {
    "aggressive": 2,
    "balanced": 4,
    "conservative": 8,
}

DEFAULT_MODE: ProtectionMode = "balanced"


@dataclass(slots=True)
class ComposedHistory:
    """Protection Zone uygulanmış geçmiş bileşeni.

    Attributes
    ----------
    compresh_md:
        Eski (koruma bölgesi dışı) mesajların TurnBox markdown formu.
        Mesaj sayısı ≤ N ise boş string.
    raw_tail:
        Koruma bölgesindeki son ≤N mesaj — provider'a raw role/content
        formatında geçirilir (sıkıştırma uygulanmaz).
    n_compressed:
        Sıkıştırılmış (TurnBox'a alınmış) mesaj sayısı.
    n_protected:
        Koruma bölgesinde tutulan mesaj sayısı (≤ N).
    mode:
        Kullanılan sıkıştırma modu.
    """

    compresh_md: str
    raw_tail: list[dict]
    n_compressed: int
    n_protected: int
    mode: ProtectionMode


def resolve_n(mode: str | int) -> int:
    """Mod string'ini N integer'a çevir.

    int doğrudan geçilirse (kullanıcı override) onu döner.
    String mod adı verilirse PROTECTION_ZONE_N tablosundan okur.
    """
    if isinstance(mode, int):
        if mode < 0:
            raise ValueError(f"Protection zone N must be ≥ 0, got {mode}")
        return mode
    if mode not in PROTECTION_ZONE_N:
        raise ValueError(
            f"Unknown protection mode {mode!r}. "
            f"Expected one of {list(PROTECTION_ZONE_N)} or an int."
        )
    return PROTECTION_ZONE_N[mode]


def compose_compresh_history(
    messages: Sequence[dict],
    turn_boxes: Sequence[TurnBox],
    *,
    upto_idx: int,
    mode: ProtectionMode | int = DEFAULT_MODE,
) -> ComposedHistory:
    """Protection Zone uygulanmış geçmiş üret.

    Parameters
    ----------
    messages:
        Tüm konuşma — her item ``{"role": str, "content": str}``.
    turn_boxes:
        ``messages`` ile aynı sırada üretilmiş TurnBox'lar.
        ``len(turn_boxes) >= upto_idx`` olmalı.
    upto_idx:
        Mevcut user mesajının indeksi (bu indekse kadar olan geçmiş
        bileşene girer; bu mesajın kendisi caller tarafından eklenir).
    mode:
        ``"aggressive"`` / ``"balanced"`` / ``"conservative"`` veya
        doğrudan integer N. Default ``"balanced"`` (N=4).

    Returns
    -------
    ComposedHistory:
        ``compresh_md`` (eski mesajların markdown'ı) +
        ``raw_tail`` (son ≤N mesaj raw).
    """
    if upto_idx < 0:
        raise ValueError(f"upto_idx must be ≥ 0, got {upto_idx}")
    if upto_idx > len(messages):
        raise ValueError(
            f"upto_idx {upto_idx} exceeds messages length {len(messages)}"
        )

    n = resolve_n(mode)
    mode_label: ProtectionMode = (
        mode if isinstance(mode, str) else "balanced"  # type: ignore[assignment]
    )

    prior_msgs = list(messages[:upto_idx])
    prior_boxes = list(turn_boxes[:upto_idx])

    if len(prior_msgs) <= n:
        # Tüm geçmiş koruma bölgesinde — sıkıştırma yok.
        return ComposedHistory(
            compresh_md="",
            raw_tail=[
                {"role": m.get("role", "user"),
                 "content": m.get("content", "") or ""}
                for m in prior_msgs
            ],
            n_compressed=0,
            n_protected=len(prior_msgs),
            mode=mode_label,
        )

    split = len(prior_msgs) - n
    compressed_boxes = prior_boxes[:split]
    raw_tail_msgs = prior_msgs[split:]

    md = render_markdown_many(compressed_boxes) if compressed_boxes else ""

    return ComposedHistory(
        compresh_md=md,
        raw_tail=[
            {"role": m.get("role", "user"),
             "content": m.get("content", "") or ""}
            for m in raw_tail_msgs
        ],
        n_compressed=split,
        n_protected=n,
        mode=mode_label,
    )


def compose_compresh_system(
    base_system: str,
    composed: ComposedHistory,
    *,
    history_preamble: str = (
        "Below is a compressed memory of the conversation so far"
        " (older turns only — the most recent turns follow as raw"
        " messages):"
    ),
    honesty_fragment: Optional[str] = None,
    empty_history_note: str = (
        "No prior conversation history."
    ),
) -> str:
    """ComposedHistory'i bir system prompt string'ine yerleştir.

    ``compresh_md`` boşsa (tüm geçmiş protection zone'da) history bloğu
    atlanır — model sadece raw tail + current user message'ı görür.
    """
    parts = [base_system.rstrip()]

    if composed.compresh_md:
        parts.append("")
        parts.append(history_preamble)
        parts.append("")
        parts.append(composed.compresh_md)
    elif composed.n_protected == 0:
        # İlk turn — geçmiş yok, koruma bölgesi yok.
        parts.append("")
        parts.append(empty_history_note)

    if honesty_fragment:
        parts.append("")
        parts.append(honesty_fragment)

    return "\n".join(parts)


def iter_protection_modes() -> Iterable[tuple[str, int]]:
    """Kayıtlı tüm protection mode'ları (paper için kolaylık)."""
    return PROTECTION_ZONE_N.items()
