"""Distribution guardrails for the emotion detector -- the permanent form of the
qualitative evaluation that caught the character-prior snowball bug, where a
running per-character history made a whole detective novel read as 80% "Cheery"
and a horror story as 62% "Fearful".

Two layers:
  1. Synthetic guards (always run): reproduce the exact pre-fix failure setup and
     assert it no longer collapses; assert detection is character-agnostic.
  2. Corpus guard (skips when the gitignored corpus is absent): sweeps real books
     and asserts no single non-neutral emotion dominates.
"""

import collections
import json
import os

import pytest

from src.emotion_pass import enrich_scene_emotions

# A non-neutral emotion covering more than this share of a book is the snowball
# signature. Post-fix real books sit in the low single digits; the pre-fix bug
# produced 62-80%.
_NON_NEUTRAL_CEILING = 0.55

_NEUTRALS = {"Flat", "Neutral"}


def _distribution(scenes):
    """Run the detector scene-by-scene (as a real book is processed) and return
    the emotion label counts aggregated across all lines."""
    counts = collections.Counter()
    for scene_id, lines in scenes:
        out, _ = enrich_scene_emotions(scene_id, lines)
        counts.update(l.get("emotion") for l in out)
    return counts


def _line(i, character, segment_type, text):
    return {"line_id": f"l{i}", "character": character, "segment_type": segment_type, "text": text}


def _mostly_neutral_book_with_one_cheery_scene(character):
    """The pre-fix trap: a recurring character whose FIRST scene has cheery cues,
    followed by many plain scenes. History bias used to paint the whole book
    cheery. With detection character-agnostic and stateless, it must not."""
    scenes = []
    # Scene 1: genuine cheery cues.
    scenes.append(("s1", [
        _line(1, character, "dialogue", "Wonderful! I am delighted and happy to see you, my friend."),
        _line(2, "Narrator", "narrative", "He smiled warmly and laughed with joy."),
    ]))
    # Scenes 2..8: plain, emotionally-flat content by the same character.
    plains = [
        "The road ran straight between the hedgerows.",
        "It is late, and the office closes at six.",
        "He opened the ledger and turned to the third column.",
        "The house stood at the corner of the second street.",
        "We took the train and arrived by noon.",
        "The document listed the dates in order.",
        "She placed the cup upon the wooden table.",
    ]
    for si, text in enumerate(plains, start=2):
        scenes.append((f"s{si}", [
            _line(si * 10, character, "dialogue", text),
            _line(si * 10 + 1, "Narrator", "narrative", "The afternoon passed without event."),
        ]))
    return scenes


def test_one_emotional_scene_does_not_snowball_the_book():
    counts = _distribution(_mostly_neutral_book_with_one_cheery_scene("Holmes"))
    total = sum(counts.values())
    non_neutral = {e: c for e, c in counts.items() if e not in _NEUTRALS}
    top = max(non_neutral.values(), default=0)
    assert top / total < _NON_NEUTRAL_CEILING, f"an emotion dominated: {dict(counts)}"
    # The plain majority must read as neutral, not inherit the opening tone.
    assert counts["Flat"] + counts["Neutral"] >= max(non_neutral.values(), default=0)


def test_distribution_identical_across_characters():
    """Swapping every character name must not change a single emotion label:
    the strongest possible statement that identity does not touch detection."""
    dist_a = _distribution(_mostly_neutral_book_with_one_cheery_scene("Holmes"))
    dist_b = _distribution(_mostly_neutral_book_with_one_cheery_scene("Moriarty"))
    assert dist_a == dist_b


# ---------------------------------------------------------------------------
# Corpus guard -- runs only when the (gitignored) pipeline corpus is present.
# ---------------------------------------------------------------------------

_CORPUS_BOOKS = [
    "The Red-Headed League",        # pre-fix: 80% Cheery
    "The Adventure of the Speckled Band",  # pre-fix: 62% Fearful
]


def _corpus_scenes(book):
    from src.book_structure_adapter import load_structure, load_line_payloads, structure_to_manifest
    manifest = structure_to_manifest(load_structure(book), line_payloads=load_line_payloads(book))
    return [
        (scene.scene_id, [l.model_dump() for l in scene.lines])
        for part in manifest.parts for chapter in part.chapters for scene in chapter.scenes
    ]


@pytest.mark.parametrize("book", _CORPUS_BOOKS)
def test_corpus_book_has_no_dominant_emotion(book):
    if not os.path.isdir(os.path.join("data", "corpus", "pipeline", book, "tier1")):
        pytest.skip(f"corpus book not present: {book}")
    counts = _distribution(_corpus_scenes(book))
    total = sum(counts.values())
    assert total > 0
    non_neutral = {e: c for e, c in counts.items() if e not in _NEUTRALS}
    for emotion, count in non_neutral.items():
        assert count / total < _NON_NEUTRAL_CEILING, (
            f"{book}: {emotion} is {count}/{total} = {count/total:.0%} of lines "
            f"(> {_NON_NEUTRAL_CEILING:.0%} ceiling) -- emotion detector is over-forcing")
