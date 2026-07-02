"""Cheap, deterministic first line of defense against prompt injection.

This is NOT the only guardrail -- the system prompt also instructs the model
to treat conversation content as untrusted and to refuse scope violations.
This module exists because that instruction-following can be probed/broken,
and a regex check that runs before any LLM call costs ~0ms and can't be
talked out of its job. It only catches blatant, high-confidence attempts;
anything subtler is left to the model's own judgment so we don't false-positive
on legitimate assessment questions that happen to share vocabulary.
"""
import re

_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above|earlier) instructions",
    r"disregard (all |the )?(previous|prior|above|earlier) instructions",
    r"you are now|from now on you are|act as (a|an) (?!hiring)",
    r"reveal (your|the) (system|hidden) prompt",
    r"what (is|are) your (system|hidden) (prompt|instructions)",
    r"print (your|the) (system|hidden) prompt",
    r"developer mode|jailbreak|dan mode",
    r"pretend (you|this) (has no|have no|are not) (restrictions|guidelines|rules)",
    r"repeat (everything|the text) (above|before this)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

REFUSAL_REPLY = (
    "I can't act on instructions embedded in the conversation like that -- I'm here to help you find "
    "SHL assessments. What role or skills are you hiring for?"
)


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)
