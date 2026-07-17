#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EPUB ingestion: spine -> chapters -> pipeline-ready text.

An EPUB's spine hands us chapter boundaries for free -- each content document
is a chapter candidate in reading order, which de-risks Loop 2 and the
front-matter scrubber entirely for this input path. This module converts an
EPUB to the plain-text shape the existing loops already parse:

    CHAPTER 1. {heading}
    {paragraphs...}

Non-content spine items (cover, nav/toc, copyright, title page) are excluded
by id/href/epub-type heuristics; the ClutterScrubber still runs downstream
(Project Gutenberg EPUBs carry the license as spine items too, and the
back-matter chop catches what the heuristics miss).
"""

import logging
import os
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger("EpubIngestion")

_NS = {
    "cn": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}
_SKIP_NAME = re.compile(r"cover|titlepage|title-page|halftitle|nav|toc|contents|"
                        r"copyright|colophon|dedication|imprint|frontmatter|backmatter|"
                        r"acknowledg|about-the|advert", re.IGNORECASE)
_BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
               "br", "tr", "section", "article"}
_SKIP_CONTENT_TAGS = {"script", "style", "head", "title"}


class _TextExtractor(HTMLParser):
    """Tag-stripping text extraction with paragraph breaks at block elements
    and the first heading captured separately (it becomes the chapter title)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.first_heading: Optional[str] = None
        self._in_heading = 0
        self._heading_buf: List[str] = []
        self._skip_depth = 0
        self._is_nav_doc = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        attrs = dict(attrs)
        if tag == "nav" and "toc" in (attrs.get("epub:type") or attrs.get("role") or ""):
            self._is_nav_doc = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
        if tag in ("h1", "h2", "h3") and self.first_heading is None:
            self._in_heading += 1
            self._heading_buf = []

    def handle_endtag(self, tag):
        if tag in _SKIP_CONTENT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in ("h1", "h2", "h3") and self._in_heading:
            self._in_heading -= 1
            heading = " ".join("".join(self._heading_buf).split())
            if heading and self.first_heading is None:
                self.first_heading = heading
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_heading:
            self._heading_buf.append(data)
        else:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r" ?\n ?", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _spine_hrefs(z: zipfile.ZipFile) -> Tuple[str, List[Tuple[str, str]]]:
    """(opf_dir, [(idref, href)]) in spine reading order."""
    container = ET.fromstring(z.read("META-INF/container.xml"))
    rootfile = container.find(".//cn:rootfile", _NS).get("full-path")
    opf_dir = posixpath.dirname(rootfile)
    opf = ET.fromstring(z.read(rootfile))
    manifest = {
        item.get("id"): (item.get("href"), item.get("media-type", ""), item.get("properties", ""))
        for item in opf.findall(".//opf:manifest/opf:item", _NS)
    }
    ordered = []
    for itemref in opf.findall(".//opf:spine/opf:itemref", _NS):
        entry = manifest.get(itemref.get("idref"))
        if not entry:
            continue
        href, media, props = entry
        if "html" not in media:
            continue
        if "nav" in (props or ""):
            continue
        ordered.append((itemref.get("idref"), posixpath.join(opf_dir, href) if opf_dir else href))
    return opf_dir, ordered


def epub_to_text(path: str, min_chapter_chars: int = 300) -> str:
    """Convert an EPUB to loop-ready plain text with CHAPTER headings.

    Spine order is chapter order. Items are dropped when their id/href names
    front/back matter, they declare themselves a nav doc, or their extracted
    text is trivially short (a cover page, an ornament page)."""
    chapters: List[Tuple[str, str]] = []
    with zipfile.ZipFile(path) as z:
        _, spine = _spine_hrefs(z)
        for idref, href in spine:
            name = f"{idref} {posixpath.basename(href)}"
            if _SKIP_NAME.search(name):
                logger.info(f"EPUB: skipping non-content spine item '{name.strip()}'.")
                continue
            try:
                doc = z.read(href).decode("utf-8", errors="replace")
            except KeyError:
                logger.warning(f"EPUB: spine href missing from archive: {href}")
                continue
            ex = _TextExtractor()
            ex.feed(doc)
            if ex._is_nav_doc:
                logger.info(f"EPUB: skipping nav/toc document '{href}'.")
                continue
            text = ex.text()
            if len(text) < min_chapter_chars:
                logger.info(f"EPUB: skipping short spine item '{href}' ({len(text)} chars).")
                continue
            heading = ex.first_heading or ""
            # the heading line also appears at the top of the body text; drop it
            if heading and text[:len(heading) + 10].strip().lower().startswith(heading.lower()):
                text = text[text.lower().find(heading.lower()) + len(heading):].lstrip()
            chapters.append((heading, text))

    if not chapters:
        raise ValueError(f"EPUB has no usable content documents: {path}")

    out = []
    for i, (heading, text) in enumerate(chapters, 1):
        # Loop 2's chapter regex anchors on CHAPTER-style lines; synthesize one
        # from the spine, keeping the document's own heading as the title.
        if re.match(r"^(chapter|letter|book|part)\b", heading, re.IGNORECASE):
            out.append(f"{heading.upper() if heading.isupper() else heading}\n\n{text}")
        else:
            out.append(f"CHAPTER {i}. {heading}\n\n{text}" if heading else f"CHAPTER {i}.\n\n{text}")
    logger.info(f"EPUB: {len(chapters)} content chapter(s) extracted from spine.")
    return "\n\n\n".join(out) + "\n"
