"""Render-path convergence: every render path (primary upload flow via
render_job, and the Console) must source lines from the same performance
script -- attribution reduction + emotion pass + character-aware delivery --
rather than raw manifest lines. These tests pin the wiring that makes Tier 1/2
performance-aware, not just Tier 3.
"""

import json
import os
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from src.production_mixer import performance_or_manifest_lines
from src.attribution_reduction import build_performance_script


class _FakeLine:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class _FakeScene:
    def __init__(self, scene_id, line_payloads):
        self.scene_id = scene_id
        self.lines = [_FakeLine(p) for p in line_payloads]


def test_performance_lines_preferred_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    book = "holmes_fixture"
    tier3 = Path("data") / "corpus" / "pipeline" / book / "tier3"
    tier3.mkdir(parents=True, exist_ok=True)
    # A performance script whose scene lines differ from the raw manifest lines.
    (tier3 / "performance_script.json").write_text(json.dumps({
        "scenes": [{
            "scene_id": "s1",
            "lines": [{"line_id": "l1", "character": "Holmes", "text": "Elementary.",
                       "emotion": "Tense", "segment_type": "dialogue"}],
        }],
    }), encoding="utf-8")

    scene = _FakeScene("s1", [{"line_id": "l1", "character": "Holmes", "text": "RAW MANIFEST TEXT"}])
    lines = performance_or_manifest_lines(book, scene)

    assert len(lines) == 1
    assert lines[0]["text"] == "Elementary."       # from performance script, not raw manifest
    assert lines[0]["emotion"] == "Tense"


def test_manifest_lines_used_when_no_performance_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = _FakeScene("s1", [{"line_id": "l1", "character": "Holmes", "text": "RAW MANIFEST TEXT"}])
    lines = performance_or_manifest_lines("no_such_book", scene)

    assert len(lines) == 1
    assert lines[0]["text"] == "RAW MANIFEST TEXT"  # graceful fallback


def _manifest(stem, scene_lines):
    return {
        "source_file": f"{stem}.txt",
        "total_parts": 1, "total_chapters": 1, "total_scenes": 1,
        "parts": [{
            "part_id": "part_p1", "title": "Part 1",
            "chapters": [{
                "chapter_id": "part_p1_c1", "title": "Chapter 1",
                "scenes": [{"scene_id": "part_p1_c1_s1", "lines": scene_lines}],
            }],
        }],
    }


def _line(line_id, character, segment_type, text):
    return {
        "line_id": line_id, "chapter": 1, "scene": 1, "line_number": 1,
        "character": character, "speaker_id": f"char_{character.lower()}",
        "segment_type": segment_type, "text": text, "emotion": "Neutral",
        "performance": {"pitch_modifier": 1.0, "speed_modifier": 1.0, "delivery_style": "neutral_narrative"},
        "post_padding_ms": 250, "attribution_method": "Tier 1 Default",
        "confidence": 1.0, "speaker_locked": True, "utterance_type": "speech",
    }


def test_render_job_builds_performance_script_for_tier1(tmp_path, monkeypatch):
    """Fix 2: run_job builds the performance script for every tier, so a plain
    Tier 1 render is performance-aware even with no prior Console crew run."""
    monkeypatch.chdir(tmp_path)

    stem = "convergence_book"
    uploads = Path("data") / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / f"{stem}.txt").write_text("placeholder", encoding="utf-8")

    # No canonical book_structure.json -> run_job takes the ingest fallback branch.
    # Stub ingest to return a manifest with a reducible attribution line.
    from src import render_job
    from src.models import ManuscriptManifest

    manifest_obj = ManuscriptManifest.model_validate(_manifest(stem, [
        _line("l1", "Holmes", "dialogue", '"I agree."'),
        _line("l2", "Narrator", "narrative", "Holmes said."),
    ]))

    monkeypatch.setitem(sys.modules, "src.llm_client",
                        types.SimpleNamespace(set_usage_context=lambda **kwargs: None))

    captured = {}

    def fake_mix_voice_track(manifest_path, out_wav, single_narrator=False):
        # By the time the mixer runs, the performance script must already exist.
        captured["perf_exists_at_mix"] = os.path.exists(
            os.path.join("data", "corpus", "pipeline", stem, "tier3", "performance_script.json"))
        return {"output": out_wav, "m4b": None, "timings": None, "acx": {"validation": {"overall_acx_compliance": "PASSED"}}}

    # Patch the real module's attributes (render_job does `from src import
    # production_mixer as pm`, which resolves the package attribute, not sys.modules).
    import src.production_mixer as _pm
    monkeypatch.setattr(_pm, "mix_voice_track", fake_mix_voice_track)
    monkeypatch.setattr(_pm, "mix_production", fake_mix_voice_track)
    monkeypatch.setattr("src.tier_1_parser.ingest_manuscript_tier_1",
                        lambda *a, **k: manifest_obj)

    os.makedirs("scratch/renders", exist_ok=True)
    os.makedirs("data/render_jobs", exist_ok=True)
    with open("data/render_jobs/jobc.json", "w", encoding="utf-8") as f:
        json.dump(_job_record(stem), f)

    exit_code = render_job.run_job("jobc")
    assert exit_code == 0

    perf_path = Path("data") / "corpus" / "pipeline" / stem / "tier3" / "performance_script.json"
    assert perf_path.exists(), "run_job must build the performance script"
    assert captured["perf_exists_at_mix"] is True, "performance script must exist before the mixer runs"

    payload = json.loads(perf_path.read_text(encoding="utf-8"))
    # The redundant 'Holmes said.' attribution line was reduced away.
    assert any(r["action"] == "remove" for r in payload["reductions"])
    # Emotion pass ran over the scene.
    assert payload["emotion_pass"]["scenes"]

    shutil.rmtree(Path("data") / "corpus" / "pipeline" / stem, ignore_errors=True)


def _job_record(stem):
    return {
        "job_id": "jobc", "book": stem, "source_file": f"data/uploads/{stem}.txt",
        "tier": 1, "owner": "local", "project_id": None, "status": "queued",
        "created_at": 0, "started_at": None, "finished_at": None, "pid": None,
        "output_wav": None, "output_m4b": None, "timings": None, "error": None,
    }
