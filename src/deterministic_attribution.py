"""Deterministic, zero-LLM speaker attribution from explicit dialogue tags.

Runs BEFORE the LLM enrichment pass so that dialogue whose speaker is *named by an
explicit tag* resolves for free -- the LLM then spends calls only on genuinely
ambiguous lines, and quota-starved scenes still get their tagged speakers.

Two high-precision signals, both built on the shared attribution grammar in
``src.attribution_reduction`` (so tag parsing stays in one place):

  1. in-line trailing tag  -- the dialogue line carries its own tag: ``"...," said Holmes``
  2. adjacent narrative tag -- a separate narrative line (``said old Mrs. Rabbit
     one morning,``) tags the quote before it (trailing) or after it (leading);
     a comma-terminated tag is an interrupted quotation and also tags the
     following fragment.

Deliberately conservative: only *named* subjects are used (pronoun tags like
"he said" and ambiguous turn-taking are left to the LLM / future coreference).
This pass sets only WHO spoke -- never emotion or delivery (detection-vs-delivery
boundary; emotion is a separate text-only pass). It is idempotent: it touches only
dialogue still on the Narrator / Tier 1 default, so it never overrides an existing
attribution and re-running it is a no-op.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.attribution_reduction import (
    _ATTRIB_SUBJ_VERB,
    _ATTRIB_VERB_SUBJ,
    _DIALOGUE_TAG_TRAIL,
    _PRONOUN_SUBJECTS,
    _parse_attribution_text,
)

# Real tag lines are often multi-sentence ("said Holmes. He leaned back."); the tag
# is the sentence touching the quote, so we test the first/last sentence, not the whole line.
# Honorific periods ("Mrs.") must not split the sentence because they are part of the name.
_SENT_SPLIT = re.compile(r"(?<!Mrs\.)(?<!Mr\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<=[.!?])\s+")
_PRONOUN_TITLES = {p.title() for p in _PRONOUN_SUBJECTS}

DETERMINISTIC_ATTRIBUTION_VERSION = "det_attr_v1"

# Explicit named tags are ground truth for who spoke -- attribute with high
# confidence and lock, so the (skipped) LLM pass and console review leave them be.
_TAG_CONFIDENCE = 0.9
# Leading determiners the attribution grammar admits into a subject ("old Mrs.
# Rabbit"); stripped so the speaker name is the character, not the phrasing.
_DETERMINERS = {"old", "young", "little", "good", "poor", "the"}


@dataclass
class AttributionStats:
    """Per-scene outcome of the deterministic pass (for logging / harness metrics)."""
    method_version: str = DETERMINISTIC_ATTRIBUTION_VERSION
    dialogue_total: int = 0
    attributed: int = 0
    by_signal: Dict[str, int] = field(default_factory=dict)
    speakers: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return round(self.attributed / self.dialogue_total, 4) if self.dialogue_total else 0.0


def _clean_name(subject: Optional[str]) -> Optional[str]:
    """Strip leading determiners; reject pronouns/empties. 'old Mrs. Rabbit' -> 'Mrs. Rabbit'."""
    if not subject or subject.lower() in _PRONOUN_SUBJECTS:
        return None
    toks = subject.split()
    while toks and toks[0].lower() in _DETERMINERS:
        toks = toks[1:]
    # Strip a trailing sentence terminator the grammar absorbed ("Holmes." -> "Holmes")
    # while keeping honorific periods, which are internal ("Mrs. Rabbit" is untouched).
    name = " ".join(toks).strip().rstrip(".").strip()
    return name or None


def _speaker_id(name: str) -> str:
    # Match the LLM path's convention exactly so alias/registry matching doesn't fork.
    return "char_" + name.lower().replace(" ", "_")


def _parse_clause_tag(text: str, *, leading: bool) -> Optional[Tuple[str, str, str]]:
    """Parse a speech tag from the first or last sentence of a narrative line.

    Real narrative tags are often multi-sentence (e.g. "said Holmes. He leaned
    back."), but only the sentence touching the quoted dialogue is the actual tag.
    For trailing tags, we inspect the first sentence; for leading tags, the last.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None

    parts = [p.strip() for p in _SENT_SPLIT.split(stripped) if p.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return _parse_attribution_text(parts[0])

    candidates = [parts[0], parts[-1]] if not leading else [parts[-1], parts[0]]
    for candidate in candidates:
        parsed = _parse_attribution_text(candidate)
        if parsed:
            return parsed
    return None


def _is_narrator_dialogue(line: Any) -> bool:
    return (
        getattr(line, "segment_type", None) == "dialogue"
        and (getattr(line, "character", "") or "Narrator") == "Narrator"
    )


def _match_roster(name: str, roster: List[str]) -> str:
    """Snap a parsed name to a known roster entry when one clearly matches (the same
    substring rule the LLM path uses), so 'Rabbit' and 'Mrs. Rabbit' don't fork."""
    low = name.lower()
    for cand in roster:
        cl = cand.lower()
        if cl == low or cl in low or low in cl:
            return cand
    return name


def _assign(line: Any, name: str, signal: str, stats: AttributionStats) -> None:
    line.character = name
    line.speaker_id = _speaker_id(name)
    line.attribution_method = f"deterministic:{signal}"
    line.confidence = _TAG_CONFIDENCE
    line.speaker_locked = True
    stats.attributed += 1
    stats.by_signal[signal] = stats.by_signal.get(signal, 0) + 1
    if name not in stats.speakers:
        stats.speakers.append(name)


def attribute_scene(lines: List[Any], roster: Optional[List[str]] = None) -> AttributionStats:
    """Attribute dialogue speakers from explicit tags. Mutates ``lines`` in place.

    Returns an :class:`AttributionStats`. Safe on any line list: a no-op when there
    are no dialogue-typed lines (e.g. offline single-voice Tier 1).
    """
    stats = AttributionStats()
    roster = list(roster or [])
    n = len(lines)
    stats.dialogue_total = sum(
        1 for l in lines if getattr(l, "segment_type", None) == "dialogue"
    )
    if not stats.dialogue_total:
        return stats

    # Signal 1: in-line trailing tag on the dialogue line itself ("...," said X).
    for l in lines:
        if not _is_narrator_dialogue(l):
            continue
        m = _DIALOGUE_TAG_TRAIL.match(l.text or "")
        if not m:
            continue
        parsed = _parse_attribution_text(m.group("tag"))
        if not parsed:
            continue
        name = _clean_name(parsed[0])
        if name:
            _assign(l, _match_roster(name, roster), "inline_tag", stats)

    # Signal 2: a separate narrative line that is an attribution tag clause. The
    # grammar is anchored (^...$), so only a bare tag matches -- ordinary narration
    # with an incidental speech verb ("Mrs. Rabbit took a basket") never does.
    # For multi-sentence lines we inspect the clause touching the quote: the first
    # sentence for trailing tags and the last sentence for leading tags.
    for i, l in enumerate(lines):
        if getattr(l, "segment_type", None) != "narrative":
            continue
        prev_line = lines[i - 1] if i - 1 >= 0 else None
        next_line = lines[i + 1] if i + 1 < n else None
        if prev_line is not None and _is_narrator_dialogue(prev_line):
            parsed = _parse_clause_tag(l.text or "", leading=False)
        elif next_line is not None and _is_narrator_dialogue(next_line):
            parsed = _parse_clause_tag(l.text or "", leading=True)
        else:
            continue
        if not parsed:
            continue
        name = _clean_name(parsed[0])
        if not name:
            continue
        name = _match_roster(name, roster)
        # A comma-terminated tag ("said X,") is a continuation: it also tags the
        # following fragment of an interrupted quotation. A period-terminated tag
        # ("said X.") is terminal and tags only the quote it trails.
        continuation = (l.text or "").rstrip().endswith(",")
        if prev_line is not None and _is_narrator_dialogue(prev_line):
            _assign(prev_line, name, "adjacent_tag", stats)
            if continuation and next_line is not None and _is_narrator_dialogue(next_line):
                _assign(next_line, name, "adjacent_tag", stats)
        elif next_line is not None and _is_narrator_dialogue(next_line):
            # Leading tag ("X said," then the quote on the next line).
            _assign(next_line, name, "adjacent_tag", stats)

    return stats
