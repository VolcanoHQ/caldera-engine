"""Scene slicing is authoritative and must match the line payloads.

Regression tests for the Peter Rabbit bug: G4 re-segmentation left the line
payloads finer (7 scenes) than loop3/the canonical structure (3 scenes), and
structure_to_manifest silently dropped the orphaned scenes (17 of 42 lines,
including the story's ending). Two guarantees now hold:

  1. The materialized structure covers every scene the payloads use (it derives
     its scene set from the authoritative post-G4 payload slice).
  2. If a structure still somehow lacks a payload scene, structure_to_manifest
     fails LOUD instead of shipping a partial book.
"""

import json
import os
from pathlib import Path

import pytest

from src.book_structure_adapter import (
    load_line_payloads,
    load_structure,
    ordered_scene_ids,
    structure_to_manifest,
)


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _line(lid, text, character="Narrator", segment_type="narrative"):
    return {
        "line_id": lid, "chapter": 1, "scene": 1, "line_number": 1,
        "character": character, "speaker_id": f"char_{character.lower()}",
        "segment_type": segment_type, "text": text,
    }


def _base_pipeline(tmp_path):
    """loop3 with only 2 (pre-G4) scenes, plus a source file the render finder sees."""
    d = tmp_path / "data" / "corpus" / "pipeline" / "bk" / "tier1"
    d.mkdir(parents=True, exist_ok=True)
    up = tmp_path / "data" / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    (up / "bk.txt").write_text("x", encoding="utf-8")
    _write(str(d / "loop1_parts.json"), [{"part_id": "part_p1", "title": "P", "text_block": "..."}])
    _write(str(d / "loop2_chapters.json"), [{"chapter_id": "part_p1_c1", "title": "C", "text_block": "..."}])
    _write(str(d / "loop3_scenes.json"), [
        {"scene_id": "part_p1_c1_s1", "text_block": "one", "boundary_source": "whole_chapter"},
        {"scene_id": "part_p1_c1_s2", "text_block": "two", "boundary_source": "whole_chapter"},
    ])
    return d


def test_structure_covers_payload_only_scenes(tmp_path, monkeypatch):
    """Enriched payloads with a finer (post-G4) slice than loop3 must still be
    fully represented in the structure and manifest -- no lines dropped."""
    monkeypatch.chdir(tmp_path)
    d = _base_pipeline(tmp_path)
    # Post-G4 enriched payload: 4 scenes, finer than loop3's 2.
    _write(str(d / "loop4_lines_enriched.json"), [
        {"scene_id": f"part_p1_c1_s{i}", "lines": [_line(f"l{i}", f"scene {i} text")]}
        for i in range(1, 5)
    ])

    st = load_structure("bk", source_file="data/uploads/bk.txt")
    assert ordered_scene_ids(st) == [f"part_p1_c1_s{i}" for i in range(1, 5)]

    manifest = structure_to_manifest(st, line_payloads=load_line_payloads("bk"))
    lines = [l for p in manifest.parts for c in p.chapters for s in c.scenes for l in s.lines]
    assert len(lines) == 4                     # every payload line survived
    # payload-only scene s4's text was reconstructed from its lines
    s4 = next(s for p in manifest.parts for c in p.chapters for s in c.scenes if s.scene_id == "part_p1_c1_s4")
    assert s4.lines and s4.lines[0].text == "scene 4 text"


def test_manifest_fails_loud_on_orphaned_payload_scene(tmp_path, monkeypatch):
    """If a payload carries a scene the structure doesn't know, raise rather than
    silently drop it (defense in depth behind the reconciliation above)."""
    monkeypatch.chdir(tmp_path)
    _base_pipeline(tmp_path)  # loop3 = 2 scenes, no loop4 -> structure has s1, s2
    st = load_structure("bk", source_file="data/uploads/bk.txt")

    payloads = [
        {"scene_id": "part_p1_c1_s1", "lines": [_line("l1", "a")]},
        {"scene_id": "part_p1_c1_s9", "lines": [_line("l9", "orphan")]},  # absent from structure
    ]
    with pytest.raises(ValueError, match="absent from the canonical structure"):
        structure_to_manifest(st, line_payloads=payloads)


def test_consistent_book_is_unchanged(tmp_path, monkeypatch):
    """When loop3 and the payloads already agree, the structure is exactly loop3's
    scenes -- the reconciliation is a no-op for well-formed books."""
    monkeypatch.chdir(tmp_path)
    d = _base_pipeline(tmp_path)
    _write(str(d / "loop4_lines_enriched.json"), [
        {"scene_id": "part_p1_c1_s1", "lines": [_line("l1", "one")]},
        {"scene_id": "part_p1_c1_s2", "lines": [_line("l2", "two")]},
    ])
    st = load_structure("bk", source_file="data/uploads/bk.txt")
    assert ordered_scene_ids(st) == ["part_p1_c1_s1", "part_p1_c1_s2"]
    manifest = structure_to_manifest(st, line_payloads=load_line_payloads("bk"))
    lines = [l for p in manifest.parts for c in p.chapters for s in c.scenes for l in s.lines]
    assert len(lines) == 2
