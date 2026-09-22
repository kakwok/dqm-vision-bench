"""
Turn the model's free-form answer into the four sections the system prompt
asks for (Instructions / Observations / Assessment / Verdict) plus a bare
GOOD/BAD verdict.

The splitting logic is ``evaluate._split_sections``; it is imported lazily
because ``evaluate`` pulls in matplotlib and the model client. If that import
is impossible (no API key in the environment) an identical local splitter is
used, so cached answers can still be structured.
"""
from __future__ import annotations

import re

SECTION_MARKERS = {
    "instructions": "Instructions",
    "observations": "Observations",
    "assessment": "Assessment",
    "verdict_text": "Verdict",
}

_VERDICT_RE = re.compile(r"\b(GOOD|BAD)\b", re.IGNORECASE)


def _split_sections_local(text: str | None, markers: dict) -> dict:
    """Same algorithm as evaluate._split_sections(keep_same_line=True)."""
    cols = list(markers.keys())
    if not text:
        return {c: "" for c in cols}
    label_to_col = {v.lower(): k for k, v in markers.items()}
    pattern = re.compile(
        r"^[#>\-\*\s]*(" + "|".join(re.escape(v) for v in markers.values()) + r")\b[:\s\*#_]*",
        re.IGNORECASE,
    )
    collected = {c: [] for c in cols}
    current = None
    for line in str(text).splitlines():
        m = pattern.match(line)
        if m:
            current = label_to_col[m.group(1).lower()]
            rest = line[m.end():].strip()
            if rest:
                collected[current].append(rest)
            continue
        if current is not None:
            collected[current].append(line)
    return {c: "\n".join(v).strip() for c, v in collected.items()}


def _split(text: str | None) -> dict:
    try:
        from evaluate import _split_sections
        return _split_sections(text, SECTION_MARKERS, keep_same_line=True)
    except Exception:
        return _split_sections_local(text, SECTION_MARKERS)


def extract_verdict(verdict_text: str, full_text: str | None = None) -> str | None:
    """GOOD/BAD from the Verdict section; fall back to the last mention in the whole text."""
    m = _VERDICT_RE.search(verdict_text or "")
    if m:
        return m.group(1).upper()
    if full_text:
        hits = _VERDICT_RE.findall(full_text)
        if hits:
            return hits[-1].upper()
    return None


def structure_response(text: str | None) -> dict:
    sections = _split(text)
    return {
        "sections": sections,
        "verdict": extract_verdict(sections.get("verdict_text", ""), text),
    }
