#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine – Human Feedback Intelligence

Records structured feedback events whenever a user corrects or overrides an
algorithmic decision.  Events are persisted as append-only JSONL so they can
be replayed, audited, and eventually fed into supervised-learning pipelines.

No model training is performed here; this module only *captures* and
*classifies* signals.

Public API
----------
record_feedback_event(...)      -> dict   persist and return the event
load_feedback_events(book_id)   -> list   read all events for a book
summarize_feedback_events(...)  -> dict   aggregate counts and stats
feedback_events_for_learning_target(target, ...) -> list  filtered view
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

# ── Storage root ──────────────────────────────────────────────────────────────
FEEDBACK_DIR = os.environ.get("CALDERA_FEEDBACK_DIR", "data/feedback")

# ── Value-score tiers ─────────────────────────────────────────────────────────
VALUE_CRITICAL = "critical"
VALUE_HIGH     = "high"
VALUE_MEDIUM   = "medium"
VALUE_LOW      = "low"

# ── Event-type classification table ──────────────────────────────────────────
#
# Maps event_type -> (learning_target, value_score)
#
# learning_target  : identifies which algorithm the correction teaches
# value_score      : priority tier for triage and downstream use
#
_EVENT_CLASSIFICATION: dict[str, tuple[str, str]] = {
    # Scene boundary corrections
    "scene_split":                  ("scene_boundary_detector",  VALUE_CRITICAL),
    "scene_merged":                 ("scene_boundary_detector",  VALUE_CRITICAL),
    "scene_boundary_corrected":     ("scene_boundary_detector",  VALUE_CRITICAL),

    # Chapter / structure detection corrections
    "chapter_boundary_corrected":   ("chapter_detector",         VALUE_CRITICAL),
    "chapter_split":                ("chapter_detector",         VALUE_CRITICAL),
    "chapter_merged":               ("chapter_detector",         VALUE_CRITICAL),

    # Speaker / attribution corrections
    "speaker_attribution_fixed":    ("speaker_attributor",       VALUE_CRITICAL),
    "speaker_locked":               ("speaker_attributor",       VALUE_CRITICAL),

    # Content-type corrections
    "content_type_changed":         ("content_type_classifier",  VALUE_CRITICAL),

    # Structure reordering
    "section_reordered":            ("structure_optimizer",      VALUE_HIGH),

    # Approval
    "approval_reversed":            ("approval_gate",            VALUE_HIGH),

    # Splits and merges that are not specifically scene/chapter
    "section_split":                ("structure_optimizer",      VALUE_HIGH),
    "sections_merged":              ("structure_optimizer",      VALUE_HIGH),

    # Removals
    "section_removed":              ("structure_optimizer",      VALUE_MEDIUM),

    # Metadata
    "metadata_corrected":           ("metadata_quality",         VALUE_MEDIUM),

    # Cosmetic / low-signal
    "section_renamed":              ("metadata_quality",         VALUE_LOW),
    "display_preference_changed":   ("ui_personalization",       VALUE_LOW),
}

# Value scores that make an event eligible for ML pipelines
_ELIGIBLE_SCORES: frozenset[str] = frozenset({VALUE_CRITICAL, VALUE_HIGH})
# Value scores that require a human reviewer before the event is used for training
_REVIEW_REQUIRED_SCORES: frozenset[str] = frozenset({VALUE_CRITICAL})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id(event_type: str, book_id: str, ts: str) -> str:
    """Deterministic-ish short ID from content hash."""
    raw = f"{event_type}:{book_id}:{ts}:{uuid.uuid4()}"
    return "evt_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _classify(
    event_type: str,
) -> tuple[str, str, bool, bool]:
    """
    Returns (learning_target, value_score, eligible_for_learning, requires_review).
    Falls back to ("general", VALUE_LOW, False, False) for unknown event types.
    """
    learning_target, value_score = _EVENT_CLASSIFICATION.get(
        event_type, ("general", VALUE_LOW)
    )
    eligible       = value_score in _ELIGIBLE_SCORES
    requires_review = value_score in _REVIEW_REQUIRED_SCORES
    return learning_target, value_score, eligible, requires_review


def _feedback_path(book_id: str) -> str:
    return os.path.join(FEEDBACK_DIR, book_id, "feedback_events.jsonl")


def _persist(book_id: str, event: dict[str, Any]) -> None:
    path = _feedback_path(book_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, default=str) + "\n")


# ── Public API ────────────────────────────────────────────────────────────────

def record_feedback_event(
    *,
    event_type: str,
    book_id: str,
    structure_version: int,
    algorithm_name: str,
    algorithm_version: str,
    before: dict[str, Any],
    after: dict[str, Any],
    confidence: float | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Persist one feedback event and return the full event dict.

    Parameters
    ----------
    event_type          : one of the keys in _EVENT_CLASSIFICATION (or any string)
    book_id             : stable book identifier (same as pipeline dir name)
    structure_version   : version counter from BookStructure at time of edit
    algorithm_name      : which algorithm produced the decision being corrected
    algorithm_version   : version string of that algorithm (e.g. "scene_v2")
    before              : snapshot of state prior to the correction
    after               : snapshot of state after the correction
    confidence          : algorithm confidence at time of original decision
    user_id             : optional opaque user identifier
    session_id          : optional session identifier
    """
    ts = _utcnow()
    learning_target, value_score, eligible, requires_review = _classify(event_type)

    event: dict[str, Any] = {
        "event_id":              _event_id(event_type, book_id, ts),
        "event_type":            event_type,
        "book_id":               book_id,
        "structure_version":     structure_version,
        "algorithm_name":        algorithm_name,
        "algorithm_version":     algorithm_version,
        "before":                before,
        "after":                 after,
        "confidence":            confidence,
        "learning_target":       learning_target,
        "value_score":           value_score,
        "eligible_for_learning": eligible,
        "requires_review":       requires_review,
        "timestamp":             ts,
        "user_id":               user_id,
        "session_id":            session_id,
    }
    _persist(book_id, event)
    return event


def load_feedback_events(book_id: str) -> list[dict[str, Any]]:
    """Load all feedback events for a book from JSONL, newest-first."""
    path = _feedback_path(book_id)
    if not os.path.exists(path):
        return []
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return list(reversed(events))  # newest first


def summarize_feedback_events(
    book_id: str | None = None,
) -> dict[str, Any]:
    """
    Aggregate feedback event statistics.

    If book_id is given, summarise that book only.
    Otherwise scan all books under FEEDBACK_DIR.
    """
    if book_id:
        all_events = load_feedback_events(book_id)
    else:
        all_events = []
        if os.path.exists(FEEDBACK_DIR):
            for bid in sorted(os.listdir(FEEDBACK_DIR)):
                all_events.extend(load_feedback_events(bid))

    by_type:   dict[str, int] = {}
    by_target: dict[str, int] = {}
    by_book:   dict[str, int] = {}
    by_value:  dict[str, int] = {}
    eligible_count = 0
    review_count   = 0

    for evt in all_events:
        _inc(by_type,   evt.get("event_type",        "unknown"))
        _inc(by_target, evt.get("learning_target",   "unknown"))
        _inc(by_book,   evt.get("book_id",           "unknown"))
        _inc(by_value,  evt.get("value_score",       "unknown"))
        if evt.get("eligible_for_learning"):
            eligible_count += 1
        if evt.get("requires_review"):
            review_count += 1

    return {
        "total_events":       len(all_events),
        "eligible_events":    eligible_count,
        "review_required":    review_count,
        "by_event_type":      by_type,
        "by_learning_target": by_target,
        "by_value_score":     by_value,
        "by_book":            by_book,
    }


def feedback_events_for_learning_target(
    target: str,
    book_id: str | None = None,
    *,
    eligible_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Return feedback events for a specific learning target.

    Parameters
    ----------
    target          : e.g. "scene_boundary_detector"
    book_id         : if given, restrict to this book
    eligible_only   : if True (default) return only events eligible for ML
    """
    if book_id:
        all_events = load_feedback_events(book_id)
    else:
        all_events = []
        if os.path.exists(FEEDBACK_DIR):
            for bid in sorted(os.listdir(FEEDBACK_DIR)):
                all_events.extend(load_feedback_events(bid))

    return [
        e for e in all_events
        if e.get("learning_target") == target
        and (not eligible_only or e.get("eligible_for_learning"))
    ]


# ── Utility ───────────────────────────────────────────────────────────────────

def _inc(d: dict[str, int], key: str) -> None:
    d[key] = d.get(key, 0) + 1


# ── Operation-to-event mapping (used by console_api) ─────────────────────────

#  Maps canonical structure operation names to the event type they produce,
#  given the content_type of the primary section being modified.
#
#  (operation, content_type)  -> event_type
#  Use content_type="*" as a wildcard fallback.
#
OPERATION_EVENT_MAP: dict[tuple[str, str], str] = {
    ("split_section",      "scene"):    "scene_split",
    ("split_section",      "chapter"):  "chapter_split",
    ("split_section",      "*"):        "section_split",
    ("merge_sections",     "scene"):    "scene_merged",
    ("merge_sections",     "chapter"):  "chapter_merged",
    ("merge_sections",     "*"):        "sections_merged",
    ("reorder_section",    "*"):        "section_reordered",
    ("remove_section",     "*"):        "section_removed",
    ("rename_section",     "*"):        "section_renamed",
    ("change_content_type","*"):        "content_type_changed",
}


def event_type_for_operation(operation: str, content_type: str) -> str:
    """Resolve the feedback event type for a structure edit operation."""
    specific = OPERATION_EVENT_MAP.get((operation, content_type))
    if specific:
        return specific
    return OPERATION_EVENT_MAP.get((operation, "*"), "structure_edit")
