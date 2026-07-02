"""Loads the scraped SHL catalog and exposes it in the shapes the rest of the
app needs:

  - `CATALOG`: list[Assessment], the full parsed catalog
  - `by_key(name_or_url)`: exact grounding lookup used to validate anything
    the LLM claims to recommend, before it's ever returned to a caller
  - `compact_catalog_text()`: a token-lean serialization fed into every LLM
    call so the model always has the entire catalog available for retrieval,
    instead of relying on a separate embedding/vector-search step

See scripts/extract_catalog.py for how data/catalog.json itself was produced
from the source PDF dump.
"""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# SHL's product-type taxonomy. The raw catalog stores full category names in
# "keys"; the API and the reference conversations both use these one-letter
# codes, so we derive them once at load time rather than hardcoding per item.
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


@dataclass(frozen=True)
class Assessment:
    entity_id: str
    name: str
    url: str
    test_type: str          # e.g. "K,S"
    job_levels: tuple[str, ...]
    languages: tuple[str, ...]
    duration: str            # human string, e.g. "25 minutes" or "" if unknown
    remote: bool
    adaptive: bool
    description: str


def _normalize(s: str) -> str:
    """Loose key for fuzzy-but-exact grounding lookups: lowercase, strip
    punctuation/whitespace differences so 'OPQ32r' and 'opq 32 r' collide,
    but two genuinely different product names never do."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load_raw(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [d for d in data if d.get("status") == "ok"]


def _to_assessment(d: dict) -> Assessment:
    codes = sorted({CATEGORY_TO_CODE.get(k, k[:1]) for k in d.get("keys", [])})
    return Assessment(
        entity_id=str(d["entity_id"]),
        name=d["name"],
        url=d["link"],
        test_type=",".join(codes) if codes else "K",
        job_levels=tuple(d.get("job_levels", [])),
        languages=tuple(d.get("languages", [])),
        duration=d.get("duration", "") or "",
        remote=(d.get("remote") == "yes"),
        adaptive=(d.get("adaptive") == "yes"),
        description=(d.get("description") or "").strip(),
    )


@lru_cache
def load_catalog(path: str) -> tuple[Assessment, ...]:
    return tuple(_to_assessment(d) for d in _load_raw(path))


@lru_cache
def _index(path: str) -> dict[str, Assessment]:
    idx: dict[str, Assessment] = {}
    for a in load_catalog(path):
        idx[_normalize(a.name)] = a
        idx[_normalize(a.url)] = a
        idx[a.entity_id] = a
    return idx


def by_key(path: str, key: str) -> Assessment | None:
    """Grounding lookup. Accepts a name, a URL, or an entity_id and returns
    the real catalog record, or None if it doesn't exist -- callers use the
    None case to drop hallucinated recommendations before they're returned."""
    if not key:
        return None
    idx = _index(path)
    return idx.get(_normalize(key)) or idx.get(key.strip())


def compact_catalog_text(path: str) -> str:
    """One line per assessment: id | name | type codes | job levels | duration
    | first-sentence description. Kept short on purpose -- this whole block
    is re-sent on every single turn, so trimming tokens here matters for
    latency under the grader's per-call timeout."""
    lines = []
    for a in load_catalog(path):
        levels = ",".join(a.job_levels) or "-"
        first_sentence = a.description.split(". ")[0][:140]
        lines.append(
            f"{a.entity_id} | {a.name} | {a.test_type} | {levels} | "
            f"{a.duration or 'untimed/unspecified'} | {first_sentence}"
        )
    return "\n".join(lines)
