from src.expression_profile import (
    EXPRESSION_STYLES,
    apply_expression_profile,
    canonicalize_expression_profile,
    canonicalize_style,
    load_expression_profiles,
)


def _line(line_id, character, segment_type, emotion):
    return {
        "line_id": line_id, "character": character, "segment_type": segment_type,
        "text": "some text", "emotion": emotion,
    }


def test_base_emotion_delivery_without_profile():
    out = apply_expression_profile([_line("l1", "Holmes", "dialogue", "Angry")])
    perf = out[0]["performance"]
    assert perf["speed_modifier"] > 1.0          # angry base speeds up
    assert perf["delivery_style"] == "angry_delivery"
    assert out[0]["expression"]["expression_style"] is None


def test_flat_and_neutral_reset_to_baseline_prosody():
    for emo in ("Flat", "Neutral"):
        out = apply_expression_profile([_line("l1", "Holmes", "dialogue", emo)])
        perf = out[0]["performance"]
        assert perf["pitch_modifier"] == 1.0
        assert perf["speed_modifier"] == 1.0
        assert perf["delivery_style"] == "neutral_narrative"


def test_restrained_damps_toward_neutral_explosive_amplifies():
    profiles = {"Holmes": {"angry": "restrained"}, "Brute": {"angry": "explosive"}}
    holmes = apply_expression_profile([_line("l1", "Holmes", "dialogue", "Angry")], profiles)[0]
    brute = apply_expression_profile([_line("l2", "Brute", "dialogue", "Angry")], profiles)[0]
    plain = apply_expression_profile([_line("l3", "Nobody", "dialogue", "Angry")], profiles)[0]

    # restrained < base < explosive on the speed deviation from 1.0
    assert holmes["performance"]["speed_modifier"] < plain["performance"]["speed_modifier"]
    assert brute["performance"]["speed_modifier"] > plain["performance"]["speed_modifier"]
    # but every one is still classified/delivered as anger, not another emotion
    assert holmes["performance"]["delivery_style"] == "angry_restrained"
    assert brute["performance"]["delivery_style"] == "angry_explosive"


def test_expression_only_applies_to_dialogue():
    """Narration is delivered by the Narrator, so a character's expression
    profile must not touch narrative lines even if the name matches."""
    profiles = {"Holmes": {"angry": "explosive"}}
    narr = apply_expression_profile([_line("l1", "Holmes", "narrative", "Angry")], profiles)[0]
    assert narr["expression"]["expression_style"] is None
    assert narr["performance"]["delivery_style"] == "angry_delivery"


def test_load_profiles_canonicalizes_emotion_nouns():
    """LLM-authored profiles may use nouns (anger/fear/joy); they fold onto the
    detector's vocabulary so profile and detector always agree."""
    profiles = load_expression_profiles([
        {"name": "Holmes", "emotion_expression_profile": {
            "anger": "restrained", "fear": "hidden", "joy": "dry", "bogus": "loud"}},
    ])
    assert profiles["Holmes"]["angry"] == "restrained"
    assert profiles["Holmes"]["fearful"] == "hidden"
    assert profiles["Holmes"]["cheery"] == "dry"
    assert "bogus" not in profiles["Holmes"]


def test_load_profiles_ignores_malformed_entries():
    assert load_expression_profiles(None) == {}
    assert load_expression_profiles([{"name": "X"}]) == {}                       # no profile field
    assert load_expression_profiles([{"emotion_expression_profile": {"angry": "x"}}]) == {}  # no name
    assert load_expression_profiles(["not a dict"]) == {}


def test_modifiers_are_clamped():
    profiles = {"X": {"violent": "explosive"}}
    out = apply_expression_profile([_line("l1", "X", "dialogue", "Violent")], profiles)[0]
    assert 0.5 <= out["performance"]["pitch_modifier"] <= 1.6
    assert 0.5 <= out["performance"]["speed_modifier"] <= 1.6


# --- Character-designer output hardening ---------------------------------

def test_canonicalize_style_aliases_and_rejects_unknown():
    assert canonicalize_style("reserved") == "restrained"
    assert canonicalize_style("VOLCANIC") == "explosive"
    assert canonicalize_style("deadpan") == "dry"
    assert canonicalize_style("restrained") == "restrained"   # already canonical
    assert canonicalize_style("gibberish") is None
    assert canonicalize_style(None) is None
    assert canonicalize_style(42) is None


def test_canonicalize_profile_folds_emotions_validates_styles_drops_junk():
    raw = {
        "anger": "stoic",       # noun emotion + alias style -> angry: restrained
        "fear": "hidden",       # noun emotion + canonical style -> fearful: hidden
        "joy": "wry",           # noun emotion + alias style -> cheery: dry
        "confusion": "loud",    # not a real emotion -> dropped
        "sad": "mysterious",    # real emotion, unknown style -> dropped
        "neutral": "measured",  # neutral isn't a deliverable emotion -> dropped
    }
    assert canonicalize_expression_profile(raw) == {
        "angry": "restrained", "fearful": "hidden", "cheery": "dry",
    }


def test_canonicalize_profile_tolerates_non_dict():
    assert canonicalize_expression_profile(None) == {}
    assert canonicalize_expression_profile("angry: restrained") == {}
    assert canonicalize_expression_profile(["angry", "restrained"]) == {}


def test_every_valid_style_is_honored_by_apply():
    """Every style the designer may emit must produce a real intensity in the
    delivery layer -- no valid style silently degrades to the no-op default."""
    for style in EXPRESSION_STYLES:
        profiles = {"X": {"angry": style}}
        out = apply_expression_profile([_line("l1", "X", "dialogue", "Angry")], profiles)[0]
        assert out["expression"]["expression_style"] == style
        # "neutral" is deliberately the 1.0 no-op; every other style must move it.
        if style != "neutral":
            assert out["performance"]["speed_modifier"] != 1.08 or style == "neutral"


def test_character_profile_validator_survives_malformed_llm_output():
    """A single character's garbage profile must not raise -- otherwise the whole
    cast's visual design is lost (design_characters returns [] on any exception)."""
    from src.scene_director import CharacterProfile, CharacterDesignSchema

    prof = CharacterProfile(
        name="X", visual_description="d",
        emotion_expression_profile={"angry": ["nested", "junk"], "fear": None, "joy": 5},
    )
    # coerced to a flat {str: str}, no exception
    assert prof.emotion_expression_profile.get("angry") == "['nested', 'junk']"
    assert "fear" not in prof.emotion_expression_profile  # None dropped

    # non-dict entirely -> empty, still valid
    prof2 = CharacterProfile(name="Y", visual_description="d", emotion_expression_profile="oops")
    assert prof2.emotion_expression_profile == {}

    # a bad profile does not sink validation of the whole cast
    cast = CharacterDesignSchema.model_validate({"profiles": [
        {"name": "Good", "visual_description": "d", "emotion_expression_profile": {"angry": "restrained"}},
        {"name": "Bad", "visual_description": "d", "emotion_expression_profile": [1, 2, 3]},
    ]})
    assert len(cast.profiles) == 2


def test_vocabulary_consistency_detection_to_expression():
    """Every emotion label the detector can emit must be understood by the
    expression layer, or a rendered line could get an emotion with no delivery."""
    import src.emotion_pass as ep
    import src.expression_profile as xp
    for title in ep._EMOTION_TITLE.values():
        assert title in xp._TITLE_TO_KEY, f"detector emits {title!r} but expression layer can't map it"
