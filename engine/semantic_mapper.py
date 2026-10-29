"""Map raw table column headers onto Parsel's target schema.

Copied from D:/parsel/engine/semantic_mapper.py at commit 9b17de6 (2026-08-21) and
trimmed to what the Spares Page Finder uses: the mapping itself, without the
unmapped-column and match-detail reporting Parsel's screens need.

Deterministic 2-tier cascade (no LLM, no ML model, offline, instant):
  1. Exact match (case-insensitive, punctuation-normalized) against schema + aliases
  2. RapidFuzz token/ratio match (edit-distance, threshold from config)
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process


def _norm(s: str) -> str:
    """lowercase, strip punctuation/whitespace for robust comparison."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class SemanticMapper:
    """Exact + fuzzy column-header mapper. Construct once, reuse across files."""

    def __init__(self, config: dict):
        self.config = config
        rules = config.get("rules", {}) or {}
        self.fuzzy_threshold = float(rules.get("fuzzy_threshold", 88))

        # Precompute alias lookup: normalized alias text -> target column
        self._alias_index = self._build_alias_index()
        # Flat list of (target_col, phrase) used for fuzzy matching
        self._phrases = self._build_phrase_list()

    # ----- index construction -------------------------------------------
    def _build_alias_index(self) -> dict[str, str]:
        idx: dict[str, str] = {}
        for col in self.config.get("target_schema", []):
            idx[_norm(col)] = col
        for col, aliases in (self.config.get("header_aliases", {}) or {}).items():
            for alias in aliases:
                idx[_norm(alias)] = col
        return idx

    def _build_phrase_list(self) -> list[tuple[str, str]]:
        phrases: list[tuple[str, str]] = []
        for col in self.config.get("target_schema", []):
            phrases.append((col, col))
        for col, aliases in (self.config.get("header_aliases", {}) or {}).items():
            for alias in aliases:
                phrases.append((col, alias))
        return phrases

    # ----- the cascade ---------------------------------------------------
    def map_columns(self, raw_columns: list[str]) -> dict[str, str]:
        """raw column name -> target schema column, for the columns that match."""
        mapping: dict[str, str] = {}
        phrase_texts = [p[1] for p in self._phrases]

        for raw in raw_columns:
            # Tier 1: exact (normalized) match
            key = _norm(raw)
            if key in self._alias_index:
                mapping[raw] = self._alias_index[key]
                continue

            # Tier 2: fuzzy match against all phrases
            best = process.extractOne(
                raw, phrase_texts, scorer=fuzz.token_sort_ratio,
            )
            if best and best[1] >= self.fuzzy_threshold:
                mapping[raw] = self._phrases[best[2]][0]

        return mapping
