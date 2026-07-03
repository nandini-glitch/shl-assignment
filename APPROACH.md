# Approach



## Provider and framework history

This went through three real iterations, not one clean design. 

**v1 — Gemini + LangChain/LangGraph.** Original build used LangGraph for
the turn-handling control flow and `langchain-google-genai` for structured
output, on the reasoning that Gemini's ~1M token context window meant the
full 377-item catalog (~23K tokens) could go into every prompt with no
retrieval step, avoiding the classic RAG failure mode of a similarity
threshold silently excluding the right item. This is architecturally sound
and is what the current version still does — but two real problems showed
up before it could be evaluated:
- Google's free tier turned out to be throttled to ~20 requests/day on this
  project, regardless of which Gemini model was used.This matters beyond local dev:
  the real grading harness hits the deployed endpoint the same way this
  eval script does, so a throttled project is a submission-blocking risk,
  not just an inconvenience.
- LangChain/LangGraph's dependency chain (`langchain-core`, `langgraph`,
  `langchain-google-genai`) turned out to be extremely version-sensitive —
  pip silently resolved to ancient, incompatible releases on a slightly
  older local Python before erroring clearly, which cost real debugging time
  for a problem that had nothing to do with the agent logic itself.

**v2 — Groq + BM25 retrieval.** Moved providers to sidestep the Gemini
throttle, since the assignment explicitly lists Groq as an accepted free
option. That immediately broke the "full catalog in every prompt" design a
different way: Groq's free tier caps a single request at 12,000 tokens
total, well under the ~23K the catalog block alone takes up. Every call
failed on turn 1. Built a BM25 keyword pre-filter (`rank-bm25`) to narrow
the catalog to a top-105 candidate set per turn before it reached the
prompt — 105 chosen empirically, not guessed, after finding that a smaller
K=60 silently excluded OPQ32r (arguably the single most broadly-relevant
personality instrument in the catalog) on a plain "stakeholder skills"
query. That was a useful, real lesson in why retrieval is risky even when
it's necessary.

**v3 — Gemini + native `google-genai` SDK, no framework, full catalog
again (current).**
 The Groq token cap was a Groq-specific problem, and the
Gemini quota throttle is an account-level problem, not a framework problem
-- so once the plan was "fix the actual account issue" rather than "route
around it with a different provider," LangChain/LangGraph stopped earning
its keep. This version drops it entirely for a plain Python function
(`app/agent.py::run_turn`) calling Google's native SDK directly with a
Pydantic `response_schema`, which is genuinely simpler and has no
version-fragile dependency chain. Dropping Groq also meant the BM25
pre-filter could come back out — Gemini's context window fits the whole
catalog, so the simpler, better-recall-ceiling design from v1 is back, this
time without the account throttle silently sabotaging it (assuming billing
is enabled or a fresh project is used -- see the evaluation section below,
this is a real open item, not resolved by this rewrite alone).

## Agent design (current)

Single function, three steps, no orchestration framework:

1. **Prefilter** (`app/guardrails.py`): regex-based prompt-injection check
   runs before any LLM call — catches "ignore previous instructions",
   "reveal your system prompt", etc. for ~0ms and can't be talked out of
   refusing. Subtler scope violations (general hiring advice, legal
   questions) are left to the model itself, which is instructed to treat
   all conversation content as untrusted input rather than instructions.
2. **Generate** (`app/agent.py::_call_model`): one structured-output call
   per turn. The model decides `intent` (clarify / recommend / refine /
   compare / refuse / finalize), `reply`, `recommended_ids` (the full
   current shortlist, not a delta), and `end_of_conversation`. One call
   instead of a multi-step pipeline is deliberate: the assignment explicitly
   warns that a non-deterministic conversation shouldn't make the system
   fall apart, and every extra LLM-to-LLM handoff is another place for that
   to happen, plus more of the 30s-per-call budget spent.
3. **Validate** (`app/agent.py::_resolve_recommendations`): every id the
   model claims is looked up against the real catalog
   (`app/catalog.py::by_key`, matched by normalized name, URL, or
   entity_id). Anything that doesn't resolve is silently dropped, never
   returned. `end_of_conversation` is only trusted when `intent == finalize`
   **and** the resolved shortlist is non-empty — a model claiming "done"
   with nothing to show doesn't get to end the conversation.

`recommendations` is empty on clarify/compare/refuse turns and the full
shortlist on recommend/refine/finalize — matching the reference
conversations' convention of only showing the table on turns that actually
establish or change it, not re-echoing it on every turn.

## Prompt design

System prompt encodes the five conversational behaviors as explicit,
mutually exclusive intents rather than leaving the model to freelance a
policy. It's told explicitly not to manufacture clarifying questions when a
request is already specific enough (turn 1 of C4 and C9 in the gold traces
recommend without any clarification, because the JD/request already had
enough signal) — over-clarifying was a real risk with a more generic "always
ask before recommending" instruction.

## Evaluation

`scripts/eval_harness.py` replays each of the 10 gold traces' user turns
against the live agent in-process, building message history from the
agent's own real replies at each step (matching how a stateless caller
actually works). It reports Recall@10 on the final shortlist per trace, an
agreement rate on *whether* a table was shown per turn (soft signal -- the
reference is one valid path, not the only one), and `end_of_conversation`
agreement on the final turn. This is scripted replay against gold user
text, not a full LLM-simulated user like the actual grading harness --
faster to iterate with, fully deterministic for regression-testing prompt
changes, at the cost of not testing how the agent handles a user who
deviates from the reference script. `tests/test_smoke.py` covers the
plumbing (grounding validation, injection short-circuit, finalize logic,
schema shape) with the LLM mocked out, so it runs without an API key.

**Open item, real and unresolved as of this rewrite:** the Gemini
project-level throttle (~20 requests/day, see the provider history above)
was never actually fixed -- development moved to Groq to route around it,
then back to Gemini for architectural simplicity once Groq's own token cap
became the binding constraint. **Before submitting, either enable billing
on the Google Cloud project (free tier: cents for this workload) or
generate a fresh project/API key, and confirm with a full
`eval_harness.py` run that all 10 traces complete without a 429.** This is
the single highest-risk item left -- a deployed endpoint that gets
throttled mid-grading fails regardless of how correct the underlying agent
logic is.

**TODO before submission:** once quota is confirmed fixed, run
`python scripts/eval_harness.py` in full and paste actual mean Recall@10 and
per-trace results here, replacing this note.

