"""What counts as a parts table, judged by its columns: the rule both readers share
(text.py for text layers, pictures.py for pictures). Column headings are recognised
with Parsel's column mapper and header score, copied into engine/.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.pdf_extractor import build_known_headers
from engine.semantic_mapper import SemanticMapper

CONFIG = json.loads((Path(__file__).parent.parent / "engine" / "config.json").read_text(encoding="utf-8"))
MAPPER = SemanticMapper(CONFIG)
KNOWN = build_known_headers(CONFIG)          # every heading Parsel knows, for its header score
CONFIDENT = float(CONFIG["options"]["score_confident"])   # a header score Parsel trusts
COLUMNS = ["Spare Part No", "Part Name", "DrawingPosNo", "Quantity"]   # the columns of a parts table

# The three answers for a page, weakest first.
NO, CHECK, SPARES = "no", "check", "spares"
strength = [NO, CHECK, SPARES].index


def signs(df) -> set[str]:
    """Which parts-table columns a table has, by Parsel's column names."""
    return set(MAPPER.map_columns([str(c) for c in df.columns]).values()) & set(COLUMNS)


def rule(found: set[str], confidence: float | None) -> str:
    """The fixed rule: used on pictures, and on text when no model has been trained.
    spares: a part-number column beside a name, item or quantity column.
    check: two of those columns without a part number (a drawing's own list, or a list
    whose "NAME OF / PARTS" header was split over two lines), or a table whose header
    Parsel is confident about (`confidence` is its header score) that fails that test,
    e.g. a parts table whose header row was misread.
    no: anything else, or no table at all (`confidence` None)."""
    if confidence is None:
        return NO
    if "Spare Part No" in found and len(found) > 1:
        return SPARES
    if len(found) > 1 or confidence >= CONFIDENT:
        return CHECK
    return NO


def bucket(df, confidence: float = 0.0) -> str:
    """The fixed rule for one table."""
    return rule(signs(df), confidence)
