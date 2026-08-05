#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.tier_1_parser import parse_tier_1_lines


def test_parse_tier1_lines_defaults_to_narrator_paragraph_flow():
    scene = (
        "Yet the old time fairy tale may now be classed as "
        "\"historical\" in the children's library.\n\n"
        "Having this thought in mind, the story of "
        "\"The Wonderful Wizard of Oz\" was written solely to please children."
    )

    lines = parse_tier_1_lines(scene_text=scene, part_num=1, chapter_num=1, scene_num=1)

    assert len(lines) == 2
    assert all(line.character == "Narrator" for line in lines)
    assert all(line.segment_type == "narrative" for line in lines)
    assert "\"historical\"" in lines[0].text
    assert "\"The Wonderful Wizard of Oz\"" in lines[1].text


def test_parse_tier1_lines_can_still_split_quotes_for_attribution_mode():
    scene = "He said, \"Come here,\" and then she replied, \"No.\""

    lines = parse_tier_1_lines(
        scene_text=scene,
        part_num=1,
        chapter_num=1,
        scene_num=1,
        narrator_only=False,
    )

    assert any(line.segment_type == "dialogue" for line in lines)
    assert any(line.segment_type == "narrative" for line in lines)
