"""L7 consolidated Character Profile — deterministic assembly of scattered signals."""

from src.character_profile import CHARACTER_PROFILE_VERSION, consolidate_profiles, count_mentions


def _designer(name, visual="a description", expr=None, inferred=False, designed_by="gemini"):
    return {
        "name": name, "visual_description": visual, "evidence_snippets": ["ev"],
        "inferred": inferred, "designed_by": designed_by,
        "emotion_expression_profile": expr or {},
    }


def _appearances(*specs):
    return [{"scene_id": sid, "position": pos, "emotion": emo} for sid, pos, emo in specs]


def test_consolidates_visual_expression_and_arc():
    index = {"Holmes": _appearances(("s1", 0, "Tense"), ("s2", 1, "Tense"), ("s4", 3, "Angry"))}
    designer = [_designer("Holmes", "gaunt detective", {"angry": "restrained"})]
    profiles = consolidate_profiles(index, designer, book_id="bk", source_work="A Scandal")
    p = profiles[0]
    assert p["profile_version"] == CHARACTER_PROFILE_VERSION
    assert p["identity"]["name"] == "Holmes"
    assert p["identity"]["role"] == "primary"                 # 3 appearances
    assert p["visual"]["visual_description"] == "gaunt detective"
    assert p["expression"] == {"angry": "restrained"}
    assert p["arc"]["dominant_emotion"] == "Tense"            # 2 tense vs 1 angry
    assert p["arc"]["emotional_range"] == ["Angry", "Tense"]
    assert [a["scene_id"] for a in p["arc"]["appearances"]] == ["s1", "s2", "s4"]
    assert p["provenance"]["book_id"] == "bk"
    assert p["provenance"]["designed_by"] == "gemini"
    assert p["voice_affinity"]["marketplace_query"] == "Holmes: gaunt detective"
    assert p["consent"] == {"shareable": False, "license": None}


def test_role_scales_with_appearances():
    index = {
        "Lead": _appearances(("s1", 0, "Flat"), ("s2", 1, "Flat"), ("s3", 2, "Flat")),
        "Bit":  _appearances(("s1", 0, "Flat")),
    }
    profiles = {p["identity"]["name"]: p for p in consolidate_profiles(index, [], book_id="b", source_work="w")}
    assert profiles["Lead"]["identity"]["role"] == "primary"
    assert profiles["Bit"]["identity"]["role"] == "supporting"


def test_mentions_elevate_a_quiet_central_character():
    """A protagonist who barely speaks but is named throughout ranks primary
    (Peter: one spoken line, mentioned constantly)."""
    index = {"Peter": _appearances(("s3", 2, "Tense"))}      # 1 speaking scene
    p = consolidate_profiles(index, [], book_id="b", source_work="w",
                             mention_counts={"Peter": 23})[0]
    assert p["identity"]["role"] == "primary"
    assert p["presence"] == {"speaking_scenes": 1, "mentions": 23}


def test_low_speaking_and_low_mentions_stays_supporting():
    index = {"Bit": _appearances(("s1", 0, "Flat"))}
    p = consolidate_profiles(index, [], book_id="b", source_work="w",
                             mention_counts={"Bit": 2})[0]
    assert p["identity"]["role"] == "supporting"


def test_count_mentions_uses_last_name_token():
    counts = count_mentions(
        ["Mrs. Rabbit", "Peter", "Narrator"],
        ["Peter ran, and Peter hid.", "Old Rabbit watched. The Rabbits fled."],
    )
    assert counts["Peter"] == 2
    assert counts["Mrs. Rabbit"] == 1     # 'Rabbit' matches once; 'Rabbits' (plural) does not
    assert counts["Narrator"] == 0


def test_designed_but_never_speaks_is_mentioned():
    """A character with a visual profile but no speaking appearances is 'mentioned'."""
    profiles = consolidate_profiles({}, [_designer("Ghost", "a pale figure")],
                                    book_id="b", source_work="w")
    assert profiles[0]["identity"]["role"] == "mentioned"
    assert profiles[0]["arc"]["appearances"] == []
    assert profiles[0]["arc"]["dominant_emotion"] == "Flat"


def test_speaks_without_a_designed_profile():
    """A speaker with no designer entry still gets a profile (empty visual)."""
    index = {"Voice": _appearances(("s1", 0, "Angry"))}
    p = consolidate_profiles(index, [], book_id="b", source_work="w")[0]
    assert p["visual"]["visual_description"] == ""
    assert p["visual"]["inferred"] is True                   # no description -> inferred
    assert p["voice_affinity"]["marketplace_query"] == "Voice audiobook character voice"
    assert p["expression"] == {}


def test_expression_is_canonicalized():
    """Noun emotions / synonym styles from the designer are folded to the canon."""
    designer = [_designer("Holmes", "d", {"anger": "stoic", "joy": "wry"})]
    p = consolidate_profiles({"Holmes": _appearances(("s1", 0, "Flat"))}, designer,
                             book_id="b", source_work="w")[0]
    assert p["expression"] == {"angry": "restrained", "cheery": "dry"}


def test_dominant_emotion_ignores_flat():
    index = {"X": _appearances(("s1", 0, "Flat"), ("s2", 1, "Flat"), ("s3", 2, "Sad"))}
    p = consolidate_profiles(index, [], book_id="b", source_work="w")[0]
    assert p["arc"]["dominant_emotion"] == "Sad"             # the only non-flat beat
