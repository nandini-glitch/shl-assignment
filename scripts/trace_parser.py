"""Parses traces/C*.md (the reference gold conversations) into a structured
form the eval harness can replay: per turn, the user's utterance, whatever
shortlist URLs the reference agent showed (if any), and the reference
end_of_conversation flag.
"""
import re
from dataclasses import dataclass
from pathlib import Path

_TURN_SPLIT = re.compile(r"^### Turn \d+\s*$", re.MULTILINE)
_URL_RE = re.compile(r"<(https://[^>]+)>")
_EOC_RE = re.compile(r"end_of_conversation.*?\*\*(true|false)\*\*", re.IGNORECASE)


@dataclass
class GoldTurn:
    user_message: str
    expected_urls: list[str]        # [] if the reference showed no table this turn
    expected_end_of_conversation: bool


def _extract_user_message(block: str) -> str:
    m = re.search(r"\*\*User\*\*\s*\n+((?:^>.*\n?)+)", block, re.MULTILINE)
    if not m:
        return ""
    lines = [ln.lstrip(">").strip() for ln in m.group(1).splitlines()]
    return " ".join(ln for ln in lines if ln)


def _extract_agent_block(block: str) -> str:
    m = re.search(r"\*\*Agent\*\*\s*\n(.*)$", block, re.DOTALL)
    return m.group(1) if m else ""


def parse_trace(path: str | Path) -> list[GoldTurn]:
    text = Path(path).read_text(encoding="utf-8")
    blocks = _TURN_SPLIT.split(text)[1:]  # [0] is the "## Conversation" preamble
    turns = []
    for block in blocks:
        user_msg = _extract_user_message(block)
        agent_block = _extract_agent_block(block)
        urls = _URL_RE.findall(agent_block)
        eoc_match = _EOC_RE.search(agent_block)
        eoc = bool(eoc_match and eoc_match.group(1).lower() == "true")
        if user_msg:
            turns.append(GoldTurn(user_message=user_msg, expected_urls=urls, expected_end_of_conversation=eoc))
    return turns


def load_all_traces(traces_dir: str | Path) -> dict[str, list[GoldTurn]]:
    traces_dir = Path(traces_dir)
    return {p.stem: parse_trace(p) for p in sorted(traces_dir.glob("*.md"))}
