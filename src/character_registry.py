#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Canonical Character Registry (Phase 1).

A cross-manuscript relational index of characters, their aliases, and the scenes
they appear in -- the entity layer the per-book profile files can't express (many
characters x many scenes x many works). It fixes character fragmentation for every
tier: the King split across "Bohemia"/"King"/"Ormstein" collapses to one canonical
character, so Tier 1's character-aware delivery, Tier 2's casting, and Tier 3's
visuals/direction all key on one identity.

DESIGN: a DERIVED INDEX. The per-book pipeline artifacts stay the source of truth;
this registry is populated from `character_profiles_consolidated.json` (via
`build_character_profiles`) and is fully rebuildable -- `ingest_book` is idempotent
(it clears the work's rows and repopulates). Standalone SQLite (stdlib only), so it
does not couple to MemPalace's vector store. See
`docs/Caldera Engine Canonical Character Registry — Design & ROI.md`.

Phase 1 = schema + writer + resolution + queries. Alias merges are applied when
present (per-book loopE artifact or console-curated); dedup value is unlocked by
those merge decisions -- the registry stores them, it does not invent them.
"""

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

DEFAULT_DB = os.path.join("data", "mempalace", "character_registry.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS works (
    work_id     TEXT PRIMARY KEY,
    title       TEXT,
    source_hash TEXT,
    consent     TEXT
);
CREATE TABLE IF NOT EXISTS characters (
    character_id   TEXT PRIMARY KEY,
    work_id        TEXT NOT NULL,
    canonical_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    work_id      TEXT NOT NULL,
    alias        TEXT NOT NULL,
    character_id TEXT NOT NULL,
    PRIMARY KEY (work_id, alias)
);
CREATE TABLE IF NOT EXISTS appearances (
    character_id TEXT NOT NULL,
    work_id      TEXT NOT NULL,
    scene_id     TEXT,
    chapter_id   TEXT,
    position     INTEGER,
    emotion      TEXT
);
CREATE TABLE IF NOT EXISTS voice_assignments (
    character_id TEXT PRIMARY KEY,
    voice_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_char_work ON characters(work_id);
CREATE INDEX IF NOT EXISTS idx_appear_char ON appearances(character_id);
"""


def load_alias_merges(loop_e: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """Flatten the loopE alias artifact ([{canonical, aliases:[...]}, ...]) into a
    {alias -> canonical} map for ingestion."""
    merges: Dict[str, str] = {}
    for entry in loop_e or []:
        if not isinstance(entry, dict):
            continue
        canonical = entry.get("canonical")
        if not canonical:
            continue
        for alias in entry.get("aliases") or []:
            if alias:
                merges[str(alias)] = str(canonical)
    return merges


def _chapter_of(scene_id: str) -> str:
    return scene_id.rsplit("_s", 1)[0] if scene_id and "_s" in scene_id else ""


class CharacterRegistry:
    def __init__(self, db_path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def character_id(work_id: str, canonical_name: str) -> str:
        """Deterministic id so the registry is rebuildable and joinable."""
        slug = lambda s: re.sub(r"\s+", "_", str(s).strip())
        return f"{slug(work_id)}::{slug(canonical_name)}"

    def ingest_book(
        self,
        work_id: str,
        title: str,
        profiles: List[Dict[str, Any]],
        *,
        merges: Optional[Dict[str, str]] = None,
        source_hash: Optional[str] = None,
        consent: Optional[str] = None,
    ) -> List[str]:
        """Populate (idempotently) the registry for one work from its consolidated
        character profiles. ``merges`` ({alias -> canonical}) collapses aliases into
        one canonical character, merging their appearances. Returns the canonical
        character names written."""
        merges = merges or {}

        # Group profiles by canonical name (applying merges), unioning appearances.
        grouped: Dict[str, Dict[str, Any]] = {}
        for p in profiles:
            name = (p.get("identity") or {}).get("name") or p.get("name")
            if not name:
                continue
            canonical = merges.get(name, name)
            g = grouped.setdefault(canonical, {"aliases": set(), "appearances": []})
            g["aliases"].add(name)
            appearances = (p.get("arc") or {}).get("appearances") or p.get("appearances") or []
            g["appearances"].extend(appearances)

        cur = self.conn.cursor()
        # Idempotent rebuild: clear this work's rows first.
        char_ids = [r[0] for r in cur.execute("SELECT character_id FROM characters WHERE work_id=?", (work_id,))]
        cur.executemany("DELETE FROM appearances WHERE character_id=?", [(c,) for c in char_ids])
        cur.executemany("DELETE FROM voice_assignments WHERE character_id=?", [(c,) for c in char_ids])
        cur.execute("DELETE FROM aliases WHERE work_id=?", (work_id,))
        cur.execute("DELETE FROM characters WHERE work_id=?", (work_id,))
        cur.execute("INSERT OR REPLACE INTO works VALUES (?,?,?,?)", (work_id, title, source_hash, consent))

        for canonical, g in grouped.items():
            cid = self.character_id(work_id, canonical)
            cur.execute("INSERT INTO characters VALUES (?,?,?)", (cid, work_id, canonical))
            for alias in g["aliases"] | {canonical}:
                cur.execute("INSERT OR REPLACE INTO aliases VALUES (?,?,?)", (work_id, alias, cid))
            for a in g["appearances"]:
                sid = a.get("scene_id", "")
                cur.execute("INSERT INTO appearances VALUES (?,?,?,?,?,?)",
                            (cid, work_id, sid, _chapter_of(sid), a.get("position"), a.get("emotion")))
        self.conn.commit()
        return sorted(grouped)

    def resolve(self, work_id: str, name: str) -> Optional[str]:
        """An attributed name (canonical or alias) -> its canonical character_id."""
        row = self.conn.execute(
            "SELECT character_id FROM aliases WHERE work_id=? AND alias=?", (work_id, name)).fetchone()
        return row[0] if row else None

    def characters_for_work(self, work_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cid, cname in self.conn.execute(
                "SELECT character_id, canonical_name FROM characters WHERE work_id=? ORDER BY canonical_name", (work_id,)):
            aliases = sorted(r[0] for r in self.conn.execute("SELECT alias FROM aliases WHERE character_id=?", (cid,)))
            n = self.conn.execute("SELECT COUNT(*) FROM appearances WHERE character_id=?", (cid,)).fetchone()[0]
            out.append({"character_id": cid, "canonical_name": cname, "aliases": aliases, "appearances": n})
        return out

    def appearances_for_character(self, character_id: str) -> List[Dict[str, Any]]:
        return [
            {"scene_id": sid, "chapter_id": chap, "position": pos, "emotion": emo, "work_id": wid}
            for (wid, sid, chap, pos, emo) in self.conn.execute(
                "SELECT work_id, scene_id, chapter_id, position, emotion FROM appearances "
                "WHERE character_id=? ORDER BY position", (character_id,))
        ]

    def set_voice_assignment(self, character_id: str, voice_ref: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO voice_assignments VALUES (?,?)", (character_id, voice_ref))
        self.conn.commit()
