"""The agent graph.

Deliberately small: three nodes, one conditional edge.

    prefilter --(injection detected)--> refuse --> END
        |
        v (clean)
     generate --> validate --> END

Two design choices worth calling out (also covered in APPROACH.md):

1. One LLM call per turn, not a multi-agent pipeline. The spec explicitly
   warns that a non-deterministic conversation shouldn't make the system
   fall apart -- every extra LLM-to-LLM handoff is another place for that to
   happen, and it also eats into the 30s-per-call / 8-turn budget. A single
   structured-output call that decides intent + reply + shortlist at once is
   simpler to reason about and cheaper to run.

2. No vector retrieval. The full compact catalog (~23k tokens) is included
   in every call instead. At 377 items this comfortably fits Gemini's
   context window, and it avoids the recall risk of a similarity threshold
   silently excluding the right assessment for a vaguely-worded query.

Grounding is enforced *after* generation, not trusted from the model: every
id the LLM claims is looked up against the real catalog, and anything that
doesn't resolve is dropped rather than passed through.
"""
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.catalog import Assessment, by_key, compact_catalog_text, load_catalog
from app.config import get_settings
from app.guardrails import REFUSAL_REPLY, looks_like_injection
from app.prompts import SYSTEM_PROMPT, build_catalog_block, build_transcript_block
from app.schemas import ChatMessage, Recommendation


class AgentTurn(BaseModel):
    """Structured output contract for the LLM. Kept separate from the public
    API schema (schemas.ChatResponse) on purpose -- this is an internal
    decision object; recommended_ids gets resolved+validated into real
    Recommendation objects before anything is returned to a caller."""

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


class GraphState(TypedDict):
    messages: list[ChatMessage]
    turn: AgentTurn | None
    recommendations: list[Recommendation]
    reply: str
    end_of_conversation: bool


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


def _prefilter_node(state: GraphState) -> GraphState:
    last_user = next((m.content for m in reversed(state["messages"]) if m.role == "user"), "")
    if looks_like_injection(last_user):
        state["reply"] = REFUSAL_REPLY
        state["recommendations"] = []
        state["end_of_conversation"] = False
        state["turn"] = AgentTurn(intent="refuse", reply=REFUSAL_REPLY)
    return state


def _route_after_prefilter(state: GraphState) -> str:
    return "END" if state["turn"] is not None else "generate"


def _make_llm(settings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.request_timeout_seconds,
    )


def _generate_node(state: GraphState) -> GraphState:
    settings = get_settings()
    catalog_text = compact_catalog_text(settings.catalog_path)
    history_text = _format_history(state["messages"])

    system = SYSTEM_PROMPT + "\n\n" + build_catalog_block(catalog_text)
    user = build_transcript_block(history_text) + (
        "\n\nRespond with your decision for this turn only, following the output contract."
    )

    llm = _make_llm(settings).with_structured_output(AgentTurn)
    turn: AgentTurn = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    state["turn"] = turn
    return state


def _validate_node(state: GraphState) -> GraphState:
    settings = get_settings()
    turn = state["turn"]
    assert turn is not None

    if turn.intent in ("recommend", "refine", "finalize"):
        recs = _resolve_recommendations(turn.recommended_ids, settings.catalog_path)
    else:
        recs = []

    state["reply"] = turn.reply
    state["recommendations"] = recs
    # end_of_conversation is only ever true on a genuine finalize with a
    # non-empty, grounded shortlist -- never trust the flag in isolation.
    state["end_of_conversation"] = bool(turn.end_of_conversation and turn.intent == "finalize" and recs)
    return state


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("prefilter", _prefilter_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("validate", _validate_node)

    graph.set_entry_point("prefilter")
    graph.add_conditional_edges(
        "prefilter", _route_after_prefilter, {"generate": "generate", "END": END}
    )
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_turn(messages: list[ChatMessage]) -> tuple[str, list[Recommendation], bool]:
    """Entry point used by the API layer. Pure function of the message
    history in, (reply, recommendations, end_of_conversation) out -- no
    hidden state, matching the stateless /chat contract."""
    settings = get_settings()
    load_catalog(settings.catalog_path)  # warm/validate cache; raises early if catalog.json is bad

    graph = get_graph()
    result: GraphState = graph.invoke(
        {
            "messages": messages,
            "turn": None,
            "recommendations": [],
            "reply": "",
            "end_of_conversation": False,
        }
    )
    return result["reply"], result["recommendations"], result["end_of_conversation"]
