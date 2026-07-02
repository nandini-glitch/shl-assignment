"""The agent's core turn logic.

No framework here on purpose -- earlier versions of this used LangGraph for
the control flow, but for a single classify-and-generate call per turn, a
plain function with an if/else is genuinely simpler and has zero extra
dependencies to version-match. LangGraph earns its keep for actual multi-node
orchestration; this isn't that.

Flow per turn:

    prefilter (regex, no LLM call) --injection detected--> refuse, return
        |
        v clean
    single structured-output call to Gemini (native google-genai SDK)
        |
        v
    validate: every recommended id is looked up against the real catalog;
    anything that doesn't resolve is dropped, never returned.

Design note on context: the full compact catalog (~23K tokens) is sent on
every call, not a retrieved subset. This was tried both ways during
development -- a BM25 pre-filter was briefly necessary when this ran on a
provider with a 12K-token-per-request cap, but Gemini's context window
(1M+ tokens) comfortably fits the whole catalog, which sidesteps the real
risk of retrieval: a similarity/keyword threshold silently excluding the
right assessment for a vaguely-worded query. Simpler and a better recall
ceiling, given the model here supports it. See APPROACH.md.
"""
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal

from app.catalog import Assessment, by_key, compact_catalog_text, load_catalog
from app.config import get_settings
from app.guardrails import REFUSAL_REPLY, looks_like_injection
from app.prompts import SYSTEM_PROMPT, build_catalog_block, build_transcript_block
from app.schemas import ChatMessage, Recommendation

# Transient-error retry budget for the single LLM call each turn makes.
# Covers short-lived hiccups (a burst 429, a flaky 503) -- it will NOT save
# you from a genuinely exhausted daily quota, which needs a plan/tier change,
# not a retry loop.
_MAX_LLM_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 5


class AgentTurn(BaseModel):
    """Structured output contract for the model. Kept separate from the
    public API schema (schemas.ChatResponse) on purpose -- this is an
    internal decision object; recommended_ids gets resolved+validated into
    real Recommendation objects before anything is returned to a caller."""

    intent: Literal["clarify", "recommend", "refine", "compare", "refuse", "finalize"]
    reply: str = Field(description="What to say to the user this turn.")
    recommended_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Catalog entity ids for the CURRENT full shortlist. Only non-empty for "
            "recommend/refine/finalize. For refine, this is the complete updated list, "
            "not just newly added items."
        ),
    )
    end_of_conversation: bool = Field(
        default=False, description="True only when intent is finalize."
    )


def _format_history(messages: list[ChatMessage]) -> str:
    lines = []
    for m in messages:
        speaker = "User" if m.role == "user" else "Agent"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def _resolve_recommendations(ids: list[str], catalog_path: str) -> list[Recommendation]:
    resolved: list[Recommendation] = []
    seen: set[str] = set()
    for raw_id in ids:
        item: Assessment | None = by_key(catalog_path, raw_id)
        if item is None or item.entity_id in seen:
            continue  # hallucinated or duplicate id -- silently dropped, never surfaced
        seen.add(item.entity_id)
        resolved.append(Recommendation(name=item.name, url=item.url, test_type=item.test_type))
        if len(resolved) == 10:
            break
    return resolved


def _make_client(settings) -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _call_model(settings, system: str, user: str) -> AgentTurn:
    """Single structured-output call, with bounded retry on transient
    errors. Isolated into its own function so tests can mock exactly this
    and nothing else."""
    client = _make_client(settings)
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=settings.llm_temperature,
        response_mime_type="application/json",
        response_schema=AgentTurn,
    )

    last_exc: Exception | None = None
    for attempt in range(_MAX_LLM_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model, contents=user, config=config
            )
            parsed = response.parsed
            if parsed is None:
                raise ValueError("model response did not match the required schema")
            return parsed
        except Exception as exc:  # noqa: BLE001 -- provider raises its own error types
            last_exc = exc
            transient = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) or "503" in str(exc)
            if not transient or attempt == _MAX_LLM_RETRIES:
                raise
            time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc  # unreachable, keeps type checkers happy


def run_turn(messages: list[ChatMessage]) -> tuple[str, list[Recommendation], bool]:
    """Entry point used by the API layer. Pure function of the message
    history in, (reply, recommendations, end_of_conversation) out -- no
    hidden state, matching the stateless /chat contract."""
    settings = get_settings()
    load_catalog(settings.catalog_path)  # warm/validate cache; raises early if catalog.json is bad

    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    if looks_like_injection(last_user):
        return REFUSAL_REPLY, [], False

    catalog_text = compact_catalog_text(settings.catalog_path)
    history_text = _format_history(messages)

    system = SYSTEM_PROMPT + "\n\n" + build_catalog_block(catalog_text)
    user = build_transcript_block(history_text) + (
        "\n\nRespond with your decision for this turn only, following the output contract."
    )

    turn = _call_model(settings, system, user)

    if turn.intent in ("recommend", "refine", "finalize"):
        recs = _resolve_recommendations(turn.recommended_ids, settings.catalog_path)
    else:
        recs = []

    # end_of_conversation is only ever trusted on a genuine finalize with a
    # non-empty, grounded shortlist -- never trust the flag in isolation.
    end_of_conversation = bool(turn.end_of_conversation and turn.intent == "finalize" and recs)
    return turn.reply, recs, end_of_conversation
