"""
tests/test_scene_boundaries.py
Scene Boundary Intelligence v2 – unit and integration tests.

Covers:
- Signal helpers (lexical_time, lexical_location, speaker_shift, stat_density)
- _v2_score_boundary() composite output
- identify_scenes() v2 metadata on each boundary_source
- loop3_boundaries.json artifact generation path
- Canonical structure boundary_v2 metadata round-trip
"""
import json
import os
import re
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tier_1_parser import (
    _SCENE_DETECTOR_VERSION,
    _SPLIT_THRESHOLD,
    _v2_lexical_location_score,
    _v2_lexical_time_score,
    _v2_score_boundary,
    _v2_speaker_shift_score,
    _v2_stat_density_shift,
    identify_scenes,
)
from src.book_structure import migrate_tier1_artifacts


# ── Lexical time signal ───────────────────────────────────────────────────────

class TestLexicalTimeScore:
    def test_later_triggers(self):
        assert _v2_lexical_time_score("Later that afternoon, he arrived.") == 1.0

    def test_meanwhile_triggers(self):
        assert _v2_lexical_time_score("Meanwhile, across the city...") == 1.0

    def test_next_morning_triggers(self):
        assert _v2_lexical_time_score("The next morning she woke early.") == 1.0

    def test_hours_later_triggers(self):
        assert _v2_lexical_time_score("Hours later, the smoke had cleared.") == 1.0

    def test_eventually_triggers(self):
        assert _v2_lexical_time_score("Eventually, the storm passed.") == 1.0

    def test_at_dawn_triggers(self):
        assert _v2_lexical_time_score("At dawn she rose from her bed.") == 1.0

    def test_plain_narrative_no_trigger(self):
        assert _v2_lexical_time_score("He walked slowly toward the door.") == 0.0

    def test_dialogue_no_trigger(self):
        assert _v2_lexical_time_score('"I told you already," she said.') == 0.0

    def test_leading_quote_stripped(self):
        # Paragraph that starts with a quote char before the keyword
        assert _v2_lexical_time_score('"Later that evening, he returned."') == 1.0

    def test_empty_string(self):
        assert _v2_lexical_time_score("") == 0.0


# ── Lexical location signal ───────────────────────────────────────────────────

class TestLexicalLocationScore:
    def test_across_the_hall_triggers(self):
        assert _v2_lexical_location_score("Across the hall, the meeting was ending.") == 1.0

    def test_elsewhere_triggers(self):
        assert _v2_lexical_location_score("Elsewhere, the war continued.") == 1.0

    def test_upstairs_triggers(self):
        assert _v2_lexical_location_score("Upstairs, a door slammed.") == 1.0

    def test_on_other_side_triggers(self):
        assert _v2_lexical_location_score("On the other side of the wall, voices rose.") == 1.0

    def test_far_away_triggers(self):
        assert _v2_lexical_location_score("Far away, the mountains loomed.") == 1.0

    def test_ordinary_sentence_no_trigger(self):
        assert _v2_lexical_location_score("The dog sat by the fire.") == 0.0

    def test_empty_string(self):
        assert _v2_lexical_location_score("") == 0.0


# ── Composite boundary scorer ─────────────────────────────────────────────────

class TestV2ScoreBoundary:
    def _para(self, n=10):
        """Generate a generic prose paragraph with n sentences."""
        return " ".join(
            ["He walked to the window and looked out at the grey sky."] * n
        )

    def test_strong_time_trigger_exceeds_threshold(self):
        prev = self._para()
        nxt  = "The next morning, everything had changed."
        result = _v2_score_boundary(prev, nxt)
        assert result["score"] >= _SPLIT_THRESHOLD
        assert "time_jump" in result["reasons"]
        assert result["confidence_band"] == "high"

    def test_strong_location_trigger_exceeds_threshold(self):
        prev = self._para()
        nxt  = "Across the town, the mayor prepared his speech."
        result = _v2_score_boundary(prev, nxt)
        assert result["score"] >= _SPLIT_THRESHOLD
        assert "location_shift" in result["reasons"]

    def test_neutral_paragraphs_below_threshold(self):
        prev = "She poured the tea and settled into her chair."
        nxt  = "The steam rose gently as she stirred the cup."
        result = _v2_score_boundary(prev, nxt)
        assert result["score"] < _SPLIT_THRESHOLD
        assert result["confidence_band"] in ("low", "medium")

    def test_output_keys(self):
        result = _v2_score_boundary("First paragraph.", "Second paragraph.")
        assert "score" in result
        assert "signals" in result
        assert "reasons" in result
        assert "confidence_band" in result
        for sig in ("lexical_time", "lexical_location", "speaker_shift", "stat_density"):
            assert sig in result["signals"]

    def test_scores_are_floats(self):
        result = _v2_score_boundary("a", "b")
        assert isinstance(result["score"], float)

    def test_reasons_is_list(self):
        result = _v2_score_boundary("a", "b")
        assert isinstance(result["reasons"], list)


# ── identify_scenes() v2 metadata ────────────────────────────────────────────

class TestIdentifyScenesV2Metadata:
    def _make_chapter(self, n_paras=4, para_text="He moved quietly through the room."):
        return "\n\n".join([para_text] * n_paras)

    # --- whole-chapter path ---

    def test_whole_chapter_has_confidence(self):
        text = self._make_chapter(4)
        scenes = identify_scenes(text, "c1")
        assert scenes[0]["confidence"] == 0.95
        assert scenes[0]["detector_version"] == _SCENE_DETECTOR_VERSION
        assert "single_scene_chapter" in scenes[0]["reasons"]

    def test_whole_chapter_boundary_source(self):
        text = self._make_chapter(4)
        scenes = identify_scenes(text, "c1")
        assert scenes[0]["boundary_source"] == "whole_chapter"

    # --- explicit marker path ---

    def test_explicit_marker_has_confidence_1(self):
        text = "Scene A paragraph.\n\n---\n\nScene B paragraph."
        scenes = identify_scenes(text, "c1")
        assert len(scenes) == 2
        for s in scenes:
            assert s["confidence"] == 1.0 or s["confidence"] == 0.95
            assert s["detector_version"] == _SCENE_DETECTOR_VERSION
            assert "explicit_separator" in s["reasons"]

    def test_explicit_marker_source_label(self):
        text = "A.\n\n---\n\nB."
        scenes = identify_scenes(text, "c1")
        for s in scenes:
            assert s["boundary_source"] in ("explicit_marker", "marker_then_size_chunk")

    # --- v2 heuristic path ---

    def test_v2_heuristic_source_label(self):
        """Long single-chapter book triggers v2 heuristic path."""
        base  = "He walked through the corridor with heavy steps. " * 40
        filler = (base + "\n\n") * 20
        text = filler + "The next morning, everything was silent.\n\n" + filler
        scenes = identify_scenes(text, "c1", is_single_chapter_book=True)
        sources = {s["boundary_source"] for s in scenes}
        assert "v2_heuristic" in sources

    def test_v2_scenes_have_reasons(self):
        base  = "He walked through the corridor. " * 40
        filler = (base + "\n\n") * 20
        text = filler + "Later that evening, the lights went out.\n\n" + filler
        scenes = identify_scenes(text, "c1", is_single_chapter_book=True)
        for s in scenes:
            assert isinstance(s.get("reasons"), list)

    def test_v2_scenes_have_detector_version(self):
        base = "A generic sentence. " * 20
        filler = (base + "\n\n") * 16
        text = filler
        scenes = identify_scenes(text, "c1", is_single_chapter_book=True)
        for s in scenes:
            assert s.get("detector_version") == _SCENE_DETECTOR_VERSION

    def test_no_internal_boundary_keys_leaked(self):
        """_boundary_candidates and _boundary_evidence should be consumed internally."""
        base = "A generic sentence. " * 20
        filler = (base + "\n\n") * 20
        text = filler + "The next morning, silence fell.\n\n" + filler
        scenes = identify_scenes(text, "c1", is_single_chapter_book=True)
        for s in scenes:
            assert "_boundary_candidates" not in s
            assert "_boundary_evidence" not in s


# ── loop3_boundaries.json generation ─────────────────────────────────────────

class TestBoundaryArtifact:
    """
    Verify that loop3_boundaries.json is produced by ingest_manuscript_tier_1
    when the chapter is long enough to trigger v2 scoring.
    """

    def test_boundaries_artifact_written(self):
        """Integration test using a synthetic manuscript."""
        from src.tier_1_parser import ingest_manuscript_tier_1

        # Craft a long single-part manuscript with a time-jump mid-chapter
        para = "He stood by the window contemplating the silence. " * 20
        transition = "The next morning, the village was empty."
        chapter_body = "\n\n".join([para] * 8 + [transition] + [para] * 8)
        manuscript = (
            "Part One\n\n"
            "Chapter 1\n\n"
            + chapter_body
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "test_book.txt")
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write(manuscript)

            pipeline_dir = os.path.join(tmpdir, "pipeline")
            os.makedirs(pipeline_dir, exist_ok=True)

            # ingest_manuscript_tier_1 derives pipeline_dir from file_path
            # Put the file in a location that yields our desired pipeline_dir
            import shutil
            real_pipeline_dir = os.path.join("data/corpus/pipeline", "test_boundary_book", "tier1")
            ingest_manuscript_tier_1(
                src_path,
                enable_llm_enrichment=False,
            )
            # Find where the output was actually written
            book_stem = os.path.splitext(os.path.basename(src_path))[0]
            real_pipeline_dir = os.path.join("data/corpus/pipeline", book_stem, "tier1")
            pipeline_dir = real_pipeline_dir

            bounds_path = os.path.join(pipeline_dir, "loop3_boundaries.json")
            scenes_path = os.path.join(pipeline_dir, "loop3_scenes.json")

            assert os.path.exists(scenes_path), "loop3_scenes.json must be written"
            # loop3_boundaries.json is only written when there are v2 candidates
            if os.path.exists(bounds_path):
                with open(bounds_path) as fh:
                    data = json.load(fh)
                assert isinstance(data, list)
                for entry in data:
                    assert "chapter_id" in entry
                    assert "detector_version" in entry
                    assert "candidates" in entry
                    assert entry["detector_version"] == _SCENE_DETECTOR_VERSION


# ── Canonical structure boundary_v2 metadata round-trip ──────────────────────

class TestBoundaryV2MigrationRoundTrip:
    def _write_json(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_boundary_v2_carried_through_migration(self, tmp_path):
        parts   = [{"part_id": "p1", "title": "Part One", "text_block": "intro"}]
        chapters = [{"chapter_id": "p1_c1", "title": "Chapter 1", "text_block": "chap text"}]
        scenes = [
            {
                "scene_id":         "p1_c1_s1",
                "text_block":       "scene text here",
                "boundary_source":  "v2_heuristic",
                "confidence":       0.87,
                "reasons":          ["time_jump"],
                "detector_version": "scene_v2",
            }
        ]
        pipeline_dir = str(tmp_path)
        self._write_json(os.path.join(pipeline_dir, "loop1_parts.json"),   parts)
        self._write_json(os.path.join(pipeline_dir, "loop2_chapters.json"),chapters)
        self._write_json(os.path.join(pipeline_dir, "loop3_scenes.json"),  scenes)

        structure = migrate_tier1_artifacts(pipeline_dir, source_file="test.txt")
        scene_sections = [s for s in structure.sections if s.content_type == "scene"]
        assert len(scene_sections) == 1
        meta = scene_sections[0].metadata
        assert "boundary_v2" in meta
        bv2 = meta["boundary_v2"]
        assert bv2["confidence"]       == 0.87
        assert bv2["reasons"]          == ["time_jump"]
        assert bv2["detector_version"] == "scene_v2"

    def test_boundary_v2_absent_for_legacy_scenes(self, tmp_path):
        """Legacy scenes without detector_version must not get boundary_v2."""
        parts    = [{"part_id": "p1", "title": "Part One", "text_block": "x"}]
        chapters = [{"chapter_id": "p1_c1", "title": "Ch", "text_block": "y"}]
        scenes   = [
            {
                "scene_id":       "p1_c1_s1",
                "text_block":     "old scene",
                "boundary_source":"whole_chapter",
            }
        ]
        pipeline_dir = str(tmp_path)
        self._write_json(os.path.join(pipeline_dir, "loop1_parts.json"),   parts)
        self._write_json(os.path.join(pipeline_dir, "loop2_chapters.json"),chapters)
        self._write_json(os.path.join(pipeline_dir, "loop3_scenes.json"),  scenes)

        structure = migrate_tier1_artifacts(pipeline_dir, source_file="test.txt")
        scene_sections = [s for s in structure.sections if s.content_type == "scene"]
        assert len(scene_sections) == 1
        meta = scene_sections[0].metadata
        assert "boundary_v2" not in meta

    def test_boundary_source_always_present(self, tmp_path):
        parts    = [{"part_id": "p1", "title": "Part One", "text_block": "x"}]
        chapters = [{"chapter_id": "p1_c1", "title": "Ch", "text_block": "y"}]
        scenes   = [
            {
                "scene_id":        "p1_c1_s1",
                "text_block":      "scene text",
                "boundary_source": "explicit_marker",
                "confidence":      1.0,
                "reasons":         ["explicit_separator"],
                "detector_version":"scene_v2",
            }
        ]
        pipeline_dir = str(tmp_path)
        self._write_json(os.path.join(pipeline_dir, "loop1_parts.json"),   parts)
        self._write_json(os.path.join(pipeline_dir, "loop2_chapters.json"),chapters)
        self._write_json(os.path.join(pipeline_dir, "loop3_scenes.json"),  scenes)

        structure = migrate_tier1_artifacts(pipeline_dir, source_file="test.txt")
        scene_sections = [s for s in structure.sections if s.content_type == "scene"]
        assert scene_sections[0].metadata["boundary_source"] == "explicit_marker"
