from src.eval_expression import score_expression


def _gold(character, profile):
    return {"character": character, "emotion_expression_profile": profile}


def _sys(name, profile):
    return {"name": name, "emotion_expression_profile": profile}


def test_exact_and_band_agreement():
    gold = [_gold("Holmes", {
        "angry":  {"style": "restrained", "confidence": 0.9, "evidence": ["e"]},
        "cheery": {"style": "dry",        "confidence": 0.8, "evidence": ["e"]},
        "fearful":{"style": "suppressed", "confidence": 0.5, "evidence": []},
    })]
    system = [_sys("Holmes", {
        "angry": "restrained",   # exact
        "cheery": "explosive",   # real disagreement (different band)
        "fearful": "restrained", # band near-miss (both reserved), not exact
    })]
    r = score_expression(gold, system)
    assert r["agreement"]["n"] == 3
    assert r["agreement"]["exact_count"] == 1          # angry only
    assert r["agreement"]["band_count"] == 2           # angry + fearful (both reserved band)


def test_coverage_reports_gold_only_and_system_only():
    gold = [_gold("Holmes", {"angry": {"style": "restrained"}}),
            _gold("Watson", {"cheery": {"style": "open"}})]
    system = [_sys("Holmes", {"angry": "restrained", "tense": "measured"})]
    r = score_expression(gold, system)
    assert ("Watson", "cheery") in r["coverage"]["gold_only"]
    assert ("Holmes", "tense") in r["coverage"]["system_only"]
    assert r["coverage"]["comparable_pairs"] == 1


def test_canonicalizes_noun_emotions_and_style_synonyms():
    """Gold authored with nouns + synonym styles must still match a canonical
    system profile -- canonicalization prevents false mismatches."""
    gold = [_gold("Holmes", {"anger": {"style": "stoic"}, "joy": {"style": "wry"}})]
    system = [_sys("Holmes", {"angry": "restrained", "cheery": "dry"})]
    r = score_expression(gold, system)
    assert r["agreement"]["n"] == 2
    assert r["agreement"]["exact_count"] == 2          # stoic->restrained, wry->dry


def test_bare_style_map_gold_shape_supported():
    """Gold may use the compact {emotion: style} form without the rich dict."""
    gold = [_gold("Holmes", {"angry": "restrained"})]
    system = [_sys("Holmes", {"angry": "restrained"})]
    r = score_expression(gold, system)
    assert r["agreement"]["exact"] == 1.0


def test_evidence_support_rate():
    gold = [_gold("Holmes", {
        "angry":  {"style": "restrained", "evidence": ["clenched jaw"]},
        "cheery": {"style": "dry",        "evidence": []},
    })]
    system = [_sys("Holmes", {"angry": "restrained", "cheery": "dry"})]
    r = score_expression(gold, system)
    assert r["evidence_support_rate"] == 0.5           # 1 of 2 gold entries has evidence


def test_calibration_matched_vs_unmatched_confidence():
    gold = [_gold("Holmes", {
        "angry":  {"style": "restrained", "confidence": 0.9},  # will match
        "cheery": {"style": "dry",        "confidence": 0.3},  # will not match
    })]
    system = [_sys("Holmes", {"angry": "restrained", "cheery": "explosive"})]
    r = score_expression(gold, system)
    assert r["calibration"]["mean_confidence_matched"] == 0.9
    assert r["calibration"]["mean_confidence_unmatched"] == 0.3


def test_no_overlap_yields_no_agreement():
    gold = [_gold("Holmes", {"angry": {"style": "restrained"}})]
    system = [_sys("Moriarty", {"angry": "explosive"})]
    r = score_expression(gold, system)
    assert r["agreement"]["n"] == 0
    assert r["agreement"]["exact"] is None
