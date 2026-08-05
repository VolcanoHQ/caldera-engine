"""
tests/test_feedback_events.py
Human Feedback Intelligence – unit and integration tests.

Covers:
- record_feedback_event() persist and return
- load_feedback_events() reads JSONL correctly
- summarize_feedback_events() aggregation
- feedback_events_for_learning_target() filtering
- _classify() value scores and learning targets
- event_type_for_operation() mapping
- JSON round-trip integrity
- Learning eligibility flags
- apply_structure_edit() integration: emits feedback event
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feedback_events import (
    VALUE_CRITICAL,
    VALUE_HIGH,
    VALUE_LOW,
    VALUE_MEDIUM,
    _classify,
    event_type_for_operation,
    feedback_events_for_learning_target,
    load_feedback_events,
    record_feedback_event,
    summarize_feedback_events,
    FEEDBACK_DIR,
)


# ── _classify() ───────────────────────────────────────────────────────────────

class TestClassify:
    def test_scene_split_is_critical(self):
        _, score, eligible, review = _classify("scene_split")
        assert score == VALUE_CRITICAL
        assert eligible is True
        assert review   is True

    def test_scene_merged_is_critical(self):
        _, score, *_ = _classify("scene_merged")
        assert score == VALUE_CRITICAL

    def test_content_type_changed_is_critical(self):
        target, score, *_ = _classify("content_type_changed")
        assert score == VALUE_CRITICAL
        assert target == "content_type_classifier"

    def test_character_profile_override_is_critical(self):
        target, score, eligible, review = _classify("character_performance_profile_corrected")
        assert target == "character_performance_profiles"
        assert score == VALUE_CRITICAL
        assert eligible is True
        assert review is True

    def test_section_reordered_is_high(self):
        _, score, eligible, review = _classify("section_reordered")
        assert score == VALUE_HIGH
        assert eligible is True
        assert review   is False

    def test_section_removed_is_medium(self):
        _, score, eligible, review = _classify("section_removed")
        assert score == VALUE_MEDIUM
        assert eligible is False

    def test_section_renamed_is_low(self):
        _, score, eligible, review = _classify("section_renamed")
        assert score == VALUE_LOW
        assert eligible is False
        assert review   is False

    def test_unknown_event_is_low(self):
        target, score, eligible, review = _classify("totally_unknown_event")
        assert score   == VALUE_LOW
        assert eligible is False


# ── event_type_for_operation() ────────────────────────────────────────────────

class TestEventTypeForOperation:
    def test_split_scene(self):
        assert event_type_for_operation("split_section", "scene")   == "scene_split"

    def test_split_chapter(self):
        assert event_type_for_operation("split_section", "chapter") == "chapter_split"

    def test_split_generic(self):
        assert event_type_for_operation("split_section", "part")    == "section_split"

    def test_merge_scene(self):
        assert event_type_for_operation("merge_sections", "scene")   == "scene_merged"

    def test_merge_chapter(self):
        assert event_type_for_operation("merge_sections", "chapter") == "chapter_merged"

    def test_reorder(self):
        assert event_type_for_operation("reorder_section", "scene")   == "section_reordered"

    def test_remove(self):
        assert event_type_for_operation("remove_section", "chapter") == "section_removed"

    def test_rename(self):
        assert event_type_for_operation("rename_section", "scene")   == "section_renamed"

    def test_change_content_type(self):
        assert event_type_for_operation("change_content_type", "scene") == "content_type_changed"

    def test_unknown_operation(self):
        result = event_type_for_operation("mystery_op", "scene")
        assert isinstance(result, str)


# ── record_feedback_event() ───────────────────────────────────────────────────

class TestRecordFeedbackEvent:
    def _record(self, tmpdir, **kwargs):
        defaults = dict(
            event_type="scene_split",
            book_id="test_book",
            structure_version=2,
            algorithm_name="scene_boundary_detector",
            algorithm_version="scene_v2",
            before={"scene_id": "s1"},
            after={"scene_id": "s1a"},
        )
        defaults.update(kwargs)
        # Point storage at tmpdir
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmpdir)
        try:
            return record_feedback_event(**defaults)
        finally:
            fe.FEEDBACK_DIR = orig

    def test_returns_dict(self, tmp_path):
        evt = self._record(tmp_path)
        assert isinstance(evt, dict)

    def test_required_keys_present(self, tmp_path):
        evt = self._record(tmp_path)
        for key in (
            "event_id", "event_type", "book_id", "structure_version",
            "algorithm_name", "algorithm_version", "before", "after",
            "learning_target", "value_score", "eligible_for_learning",
            "requires_review", "timestamp",
        ):
            assert key in evt, f"Missing key: {key}"

    def test_event_id_is_unique(self, tmp_path):
        e1 = self._record(tmp_path)
        e2 = self._record(tmp_path)
        assert e1["event_id"] != e2["event_id"]

    def test_event_persisted_to_jsonl(self, tmp_path):
        import src.feedback_events as fe
        fe_dir = str(tmp_path)
        self._record(tmp_path)
        path = os.path.join(fe_dir, "test_book", "feedback_events.jsonl")
        assert os.path.exists(path)
        with open(path) as fh:
            line = fh.readline().strip()
        data = json.loads(line)
        assert data["event_type"] == "scene_split"

    def test_confidence_optional(self, tmp_path):
        evt = self._record(tmp_path, confidence=0.72)
        assert evt["confidence"] == 0.72

    def test_confidence_none_by_default(self, tmp_path):
        evt = self._record(tmp_path)
        assert evt["confidence"] is None

    def test_eligible_for_learning_scene_split(self, tmp_path):
        evt = self._record(tmp_path, event_type="scene_split")
        assert evt["eligible_for_learning"] is True
        assert evt["requires_review"]       is True

    def test_not_eligible_for_rename(self, tmp_path):
        evt = self._record(tmp_path, event_type="section_renamed")
        assert evt["eligible_for_learning"] is False
        assert evt["requires_review"]       is False


# ── load_feedback_events() ────────────────────────────────────────────────────

class TestLoadFeedbackEvents:
    def _seed(self, tmpdir, book_id="mybook", n=3, event_type="scene_split"):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmpdir)
        try:
            for _ in range(n):
                record_feedback_event(
                    event_type=event_type,
                    book_id=book_id,
                    structure_version=1,
                    algorithm_name="alg",
                    algorithm_version="v1",
                    before={},
                    after={},
                )
        finally:
            fe.FEEDBACK_DIR = orig

    def test_returns_list(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed(tmp_path)
            events = load_feedback_events("mybook")
        finally:
            fe.FEEDBACK_DIR = orig
        assert isinstance(events, list)

    def test_count_matches(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed(tmp_path, n=5)
            events = load_feedback_events("mybook")
        finally:
            fe.FEEDBACK_DIR = orig
        assert len(events) == 5

    def test_empty_for_unknown_book(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            result = load_feedback_events("nonexistent")
        finally:
            fe.FEEDBACK_DIR = orig
        assert result == []

    def test_newest_first_ordering(self, tmp_path):
        """load_feedback_events should return most recent events first."""
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed(tmp_path, n=3)
            events = load_feedback_events("mybook")
        finally:
            fe.FEEDBACK_DIR = orig
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps, reverse=True)


# ── summarize_feedback_events() ───────────────────────────────────────────────

class TestSummarizeFeedbackEvents:
    def _seed_book(self, tmpdir, book_id, events):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmpdir)
        try:
            for evt_type in events:
                record_feedback_event(
                    event_type=evt_type,
                    book_id=book_id,
                    structure_version=1,
                    algorithm_name="alg",
                    algorithm_version="v1",
                    before={},
                    after={},
                )
        finally:
            fe.FEEDBACK_DIR = orig

    def test_total_events(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed_book(tmp_path, "book1", ["scene_split", "scene_split", "section_renamed"])
            result = summarize_feedback_events("book1")
        finally:
            fe.FEEDBACK_DIR = orig
        assert result["total_events"] == 3

    def test_eligible_count(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed_book(tmp_path, "book1", ["scene_split", "section_renamed", "section_reordered"])
            result = summarize_feedback_events("book1")
        finally:
            fe.FEEDBACK_DIR = orig
        # scene_split (critical) + section_reordered (high) = 2 eligible
        assert result["eligible_events"] == 2
        assert result["review_required"] == 1  # only critical

    def test_by_event_type_keys(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed_book(tmp_path, "book1", ["scene_split", "section_renamed"])
            result = summarize_feedback_events("book1")
        finally:
            fe.FEEDBACK_DIR = orig
        assert "scene_split"     in result["by_event_type"]
        assert "section_renamed" in result["by_event_type"]

    def test_empty_book_returns_zeros(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            result = summarize_feedback_events("nonexistent")
        finally:
            fe.FEEDBACK_DIR = orig
        assert result["total_events"] == 0


# ── feedback_events_for_learning_target() ────────────────────────────────────

class TestFeedbackEventsForLearningTarget:
    def _seed(self, tmpdir, book_id, event_types):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmpdir)
        try:
            for et in event_types:
                record_feedback_event(
                    event_type=et,
                    book_id=book_id,
                    structure_version=1,
                    algorithm_name="alg",
                    algorithm_version="v1",
                    before={},
                    after={},
                )
        finally:
            fe.FEEDBACK_DIR = orig

    def test_filters_by_target(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed(tmp_path, "b1", ["scene_split", "section_renamed", "scene_merged"])
            results = feedback_events_for_learning_target("scene_boundary_detector", "b1")
        finally:
            fe.FEEDBACK_DIR = orig
        assert len(results) == 2
        for r in results:
            assert r["learning_target"] == "scene_boundary_detector"

    def test_eligible_only_filter(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed(tmp_path, "b1", ["scene_split", "section_renamed"])
            # eligible_only=False includes all
            all_results = feedback_events_for_learning_target(
                "metadata_quality", "b1", eligible_only=False
            )
            eligible_results = feedback_events_for_learning_target(
                "metadata_quality", "b1", eligible_only=True
            )
        finally:
            fe.FEEDBACK_DIR = orig
        # section_renamed → metadata_quality / VALUE_LOW (not eligible)
        assert len(all_results) >= 1
        assert len(eligible_results) == 0

    def test_empty_when_no_match(self, tmp_path):
        import src.feedback_events as fe
        orig = fe.FEEDBACK_DIR
        fe.FEEDBACK_DIR = str(tmp_path)
        try:
            self._seed(tmp_path, "b1", ["section_renamed"])
            results = feedback_events_for_learning_target("scene_boundary_detector", "b1")
        finally:
            fe.FEEDBACK_DIR = orig
        assert results == []


# ── apply_structure_edit() integration ───────────────────────────────────────

class TestApplyStructureEditFeedbackIntegration:
    """
    Verify that apply_structure_edit() automatically emits feedback events,
    using the real Frankenstein book that exists in the test pipeline.
    """

    REAL_BOOK = "Frankenstein"

    def _load_frankenstein(self):
        """Load Frankenstein structure; skip if not present."""
        import src.book_structure_adapter as bsa
        try:
            return bsa.load_structure(self.REAL_BOOK)
        except Exception:
            return None

    def test_feedback_event_emitted_on_rename(self, tmp_path, monkeypatch):
        """Rename a section in Frankenstein and verify a feedback event is emitted."""
        import src.feedback_events as fe
        import src.console_api as ca

        structure = self._load_frankenstein()
        if structure is None:
            pytest.skip("Frankenstein test book not available")

        # Direct the feedback output to a temp dir so tests remain isolated
        monkeypatch.setattr(fe, "FEEDBACK_DIR", str(tmp_path / "feedback"))

        scene_section = next(
            (s for s in structure.sections if s.content_type == "scene"), None
        )
        if scene_section is None:
            pytest.skip("No scene sections found in Frankenstein")

        result = ca.apply_structure_edit(
            self.REAL_BOOK,
            "rename_section",
            {"section_id": scene_section.section_id, "title": scene_section.title},
        )
        # apply_structure_edit returns None only for bad book; should succeed
        assert result is not None

        events = fe.load_feedback_events(self.REAL_BOOK)
        assert len(events) >= 1
        rename_evts = [e for e in events if e["event_type"] == "section_renamed"]
        assert len(rename_evts) >= 1

    def test_feedback_event_has_required_keys(self, tmp_path, monkeypatch):
        """Check that feedback event schema has all required keys."""
        import src.feedback_events as fe
        import src.console_api as ca

        structure = self._load_frankenstein()
        if structure is None:
            pytest.skip("Frankenstein test book not available")

        monkeypatch.setattr(fe, "FEEDBACK_DIR", str(tmp_path / "feedback"))

        scene_section = next(
            (s for s in structure.sections if s.content_type == "scene"), None
        )
        if scene_section is None:
            pytest.skip("No scene sections found")

        ca.apply_structure_edit(
            self.REAL_BOOK,
            "rename_section",
            {"section_id": scene_section.section_id, "title": scene_section.title},
        )

        events = fe.load_feedback_events(self.REAL_BOOK)
        assert events
        evt = events[0]
        for key in (
            "event_id", "event_type", "book_id", "structure_version",
            "learning_target", "value_score", "eligible_for_learning",
            "requires_review", "timestamp", "before", "after",
        ):
            assert key in evt, f"Missing key {key!r} in feedback event"
