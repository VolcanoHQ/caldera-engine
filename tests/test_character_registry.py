"""Canonical Character Registry (Phase 1) — cross-manuscript character index."""

from src.character_registry import CharacterRegistry, load_alias_merges


def _profile(name, appearances):
    return {"identity": {"name": name},
            "arc": {"appearances": [{"scene_id": s, "position": p, "emotion": e} for s, p, e in appearances]}}


def _registry(tmp_path):
    return CharacterRegistry(str(tmp_path / "reg.db"))


def test_ingest_and_query_within_work(tmp_path):
    reg = _registry(tmp_path)
    profiles = [
        _profile("Holmes", [("part_p1_c1_s1", 0, "Tense"), ("part_p1_c2_s1", 3, "Angry")]),
        _profile("Watson", [("part_p1_c1_s1", 1, "Flat")]),
    ]
    reg.ingest_book("A Scandal", "A Scandal in Bohemia", profiles)
    chars = {c["canonical_name"]: c for c in reg.characters_for_work("A Scandal")}
    assert set(chars) == {"Holmes", "Watson"}
    assert chars["Holmes"]["appearances"] == 2
    # appearances carry chapter derived from the scene id
    cid = reg.resolve("A Scandal", "Holmes")
    apps = reg.appearances_for_character(cid)
    assert [a["chapter_id"] for a in apps] == ["part_p1_c1", "part_p1_c2"]


def test_alias_merges_collapse_one_character_and_union_appearances(tmp_path):
    """The King, split across Bohemia/King/Ormstein, becomes ONE canonical
    character with all appearances merged -- the Tier-1/2/3 fix."""
    reg = _registry(tmp_path)
    profiles = [
        _profile("King", [("part_p1_c1_s2", 2, "Angry")]),
        _profile("Bohemia", [("part_p1_c1_s3", 3, "Sad"), ("part_p1_c1_s4", 4, "Tense")]),
        _profile("Ormstein", [("part_p1_c2_s1", 5, "Flat")]),
        _profile("Holmes", [("part_p1_c1_s1", 0, "Tense")]),
    ]
    merges = load_alias_merges([{"canonical": "King", "aliases": ["Bohemia", "Ormstein"]}])
    reg.ingest_book("A Scandal", "A Scandal in Bohemia", profiles, merges=merges)

    chars = {c["canonical_name"]: c for c in reg.characters_for_work("A Scandal")}
    assert set(chars) == {"King", "Holmes"}                       # 4 names -> 2 characters
    king = chars["King"]
    assert set(king["aliases"]) == {"King", "Bohemia", "Ormstein"}
    assert king["appearances"] == 4                              # 1 + 2 + 1 merged

    # every alias resolves to the same canonical character
    kid = reg.resolve("A Scandal", "King")
    assert reg.resolve("A Scandal", "Bohemia") == kid
    assert reg.resolve("A Scandal", "Ormstein") == kid


def test_ingest_is_idempotent_and_rebuildable(tmp_path):
    reg = _registry(tmp_path)
    profiles = [_profile("Holmes", [("part_p1_c1_s1", 0, "Tense")])]
    reg.ingest_book("W", "W", profiles)
    reg.ingest_book("W", "W", profiles)                          # re-run
    chars = reg.characters_for_work("W")
    assert len(chars) == 1 and chars[0]["appearances"] == 1     # no duplication


def test_reingest_reflects_updated_merges(tmp_path):
    reg = _registry(tmp_path)
    profiles = [_profile("King", [("s1", 0, "Flat")]), _profile("Bohemia", [("s2", 1, "Flat")])]
    reg.ingest_book("W", "W", profiles)                          # no merges: 2 characters
    assert len(reg.characters_for_work("W")) == 2
    reg.ingest_book("W", "W", profiles, merges={"Bohemia": "King"})   # now merged
    chars = reg.characters_for_work("W")
    assert len(chars) == 1 and chars[0]["appearances"] == 2


def test_resolve_unknown_name_returns_none(tmp_path):
    reg = _registry(tmp_path)
    reg.ingest_book("W", "W", [_profile("Holmes", [("s1", 0, "Flat")])])
    assert reg.resolve("W", "Moriarty") is None


def test_cross_work_characters_are_distinct(tmp_path):
    """Phase 1 keeps identity per-work: same name in two works = two entities."""
    reg = _registry(tmp_path)
    reg.ingest_book("BookA", "A", [_profile("Peter", [("s1", 0, "Flat")])])
    reg.ingest_book("BookB", "B", [_profile("Peter", [("s1", 0, "Flat")])])
    assert reg.resolve("BookA", "Peter") != reg.resolve("BookB", "Peter")
