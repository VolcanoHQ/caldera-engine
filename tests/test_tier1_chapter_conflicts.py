#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.tier_1_parser import identify_chapters


def _block(text: str) -> str:
    return (text + " ") * 80


def test_identify_chapters_renames_intro_like_duplicate_chapter_one():
    part_text = (
        "CHAPTER 1. Introduction\n\n"
        + _block("This introduction prepares the reader for the tale.")
        + "\n\nCHAPTER I\n\n"
        + _block("Dorothy lived in the midst of the great Kansas prairies.")
    )

    chapters = identify_chapters(part_text, "part_p1")

    assert len(chapters) == 2
    assert chapters[0]["title"] == "Introduction"
    assert chapters[1]["title"] == "Chapter 1"


def test_identify_chapters_merges_near_duplicate_same_number_blocks():
    body = _block("Alice was beginning to get very tired of sitting by her sister on the bank.")
    part_text = (
        "CHAPTER I\n\n"
        + body
        + "\n\nCHAPTER 1\n\n"
        + body
    )

    chapters = identify_chapters(part_text, "part_p1")

    assert len(chapters) == 1
    assert chapters[0]["title"] == "Chapter 1"


def test_identify_chapters_normalizes_roman_to_arabic_title():
    part_text = (
        "CHAPTER IV. The Road Through the Forest\n\n"
        + _block("They followed the yellow brick road all afternoon.")
    )

    chapters = identify_chapters(part_text, "part_p1")

    assert len(chapters) == 1
    assert chapters[0]["title"] == "Chapter 4: The Road Through the Forest"
