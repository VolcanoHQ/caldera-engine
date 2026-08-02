"""Tests for the deterministic (zero-LLM) tag-based speaker attribution pass."""
import itertools

from src.deterministic_attribution import (
    DETERMINISTIC_ATTRIBUTION_VERSION,
    attribute_scene,
)
from src.models import ScriptLine

_ids = itertools.count(1)


def _line(text, segment_type, character="Narrator"):
    n = next(_ids)
    return ScriptLine(
        line_id=f"l{n:04d}",
        chapter=1,
        scene=1,
        line_number=n,
        character=character,
        speaker_id="char_narrator" if character == "Narrator" else f"char_{character.lower()}",
        segment_type=segment_type,
        text=text,
    )


def _dlg(text, character="Narrator"):
    return _line(text, "dialogue", character)


def _nar(text):
    return _line(text, "narrative")


def test_inline_trailing_tag():
    lines = [_dlg('"Good evening," said Holmes.')]
    stats = attribute_scene(lines)
    assert lines[0].character == "Holmes"
    assert lines[0].speaker_id == "char_holmes"
    assert lines[0].attribution_method == "deterministic:inline_tag"
    assert lines[0].confidence == 0.9
    assert lines[0].speaker_locked is True
    assert stats.attributed == 1
    assert stats.by_signal == {"inline_tag": 1}
    assert stats.coverage == 1.0
    assert stats.method_version == DETERMINISTIC_ATTRIBUTION_VERSION


def test_adjacent_trailing_tag():
    lines = [_dlg('"Good evening."'), _nar("said Holmes.")]
    attribute_scene(lines)
    assert lines[0].character == "Holmes"
    assert lines[0].attribution_method == "deterministic:adjacent_tag"


def test_multisentence_trailing_tag_uses_first_sentence():
    lines = [_dlg('"Good evening."'), _nar("said Holmes. He leaned back.")]
    attribute_scene(lines)
    assert lines[0].character == "Holmes"


def test_multisentence_leading_tag_uses_last_sentence():
    lines = [_nar("He leaned back. Holmes said,"), _dlg('"Good evening."')]
    attribute_scene(lines)
    assert lines[1].character == "Holmes"


def test_leading_tag_attributes_following_quote():
    lines = [_nar("Watson said,"), _dlg('"Indeed, most curious."')]
    attribute_scene(lines)
    assert lines[1].character == "Watson"


def test_interrupted_quotation_tags_both_fragments():
    # The Peter Rabbit pattern: a comma-terminated tag between two fragments.
    lines = [
        _dlg('"Now my dears,"'),
        _nar("said old Mrs. Rabbit one morning,"),
        _dlg('"you may go into the fields."'),
    ]
    attribute_scene(lines)
    assert lines[0].character == "Mrs. Rabbit"  # determiner "old" stripped
    assert lines[2].character == "Mrs. Rabbit"


def test_terminal_tag_does_not_bleed_to_next_quote():
    # "said Alice." is terminal -> tags only the quote before it, not Bob's line.
    lines = [
        _dlg('"Hello,"'),
        _nar("said Alice."),
        _dlg('"Goodbye,"'),
        _nar("said Bob."),
    ]
    attribute_scene(lines)
    assert lines[0].character == "Alice"
    assert lines[2].character == "Bob"


def test_pronoun_tag_left_for_llm():
    lines = [_dlg('"Hello there."'), _nar("he said.")]
    attribute_scene(lines)
    assert lines[0].character == "Narrator"  # pronoun -> not resolved deterministically


def test_ordinary_narration_is_not_a_tag():
    # No speech verb -> no attribution, even though it names a character.
    lines = [_nar("Mrs. Rabbit took a large basket."), _dlg('"Hello."')]
    stats = attribute_scene(lines)
    assert lines[1].character == "Narrator"
    assert stats.attributed == 0


def test_roster_snapping_prevents_forking():
    lines = [_dlg('"Hello."'), _nar("said Rabbit.")]
    attribute_scene(lines, roster=["Mrs. Rabbit"])
    assert lines[0].character == "Mrs. Rabbit"


def test_does_not_override_existing_attribution():
    lines = [_dlg('"Hello," said Holmes.', character="Watson")]
    stats = attribute_scene(lines)
    assert lines[0].character == "Watson"  # already attributed -> untouched
    assert stats.attributed == 0


def test_no_dialogue_is_noop():
    lines = [_nar("It was a dark and stormy night.")]
    stats = attribute_scene(lines)
    assert stats.dialogue_total == 0
    assert stats.attributed == 0
    assert stats.coverage == 0.0


def test_uncapitalized_common_noun_speaker_left_for_llm():
    # "the cryptographer" is not a capitalized name -> the grammar doesn't treat it
    # as a deterministic tag; role/common-noun speakers are left to the LLM.
    lines = [_dlg('"E," said the cryptographer.')]
    attribute_scene(lines)
    assert lines[0].character == "Narrator"
