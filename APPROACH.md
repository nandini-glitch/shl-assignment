# Approach

## Catalog

The provided catalog link resolved to a browser JSON viewer printed to PDF — 242
pages, with the viewer's floating "Pretty print" button overlapping the first
line of every page and long strings soft-wrapped into illegal literal newlines.
`scripts/extract_catalog.py` fixes both (filters characters by exact y-position
to drop the overlay; re-scans the text tracking JSON-string state to undo the
wrapping) and reproduces `data/catalog.json` deterministically — 377 assessments,
all `status: ok`, zero data loss, verified byte-identical against a manual
extraction. This mattered more than it sounds: a naive `pdftotext` dump parses
as garbage, and treating that as unfixable would have meant either giving up
20%+ of the catalog or building a much shakier regex-based partial parser.

## Retrieval: no vector store

At 377 items, the compact serialization of the whole catalog (id, name, type
codes, job levels, duration, one-sentence description) is ~23k tokens — well
inside Gemini's context window. Rather than embed the catalog and do
similarity search, every `/chat` call sends the full compact catalog to the
model directly. Trade-off, made deliberately:

- **Pro:** no similarity-threshold tuning, no risk of a correct-but-lexically-
  distant assessment getting excluded before the model ever sees it — the
  single biggest recall-killer in a small-catalog RAG setup.
- **Con:** ~23k tokens of input on every turn, which costs latency and money
  at this catalog size, and would stop scaling cleanly if SHL's real catalog
  (thousands of items) were the target instead of the ~377-item subset here.

If the catalog were 10x larger, I'd add a cheap keyword/BM25 pre-filter to
shrink the candidate set before the LLM call, and keep this design as the
fallback for genuinely vague queries where keyword filtering would exclude
the right answer. Not needed at this scale.

## Agent design

One LangGraph graph, three nodes: `prefilter -> generate -> validate`.

- **prefilter**: regex-based prompt-injection check, runs before any LLM
  call. Catches the obvious cases ("ignore previous instructions", "reveal
  your system prompt", "you are now DAN") for ~0ms and can't be talked out of
  refusing. Subtler scope violations (general hiring advice, legal questions)
  are left to the model, which is instructed to treat all conversation
  content as untrusted input.
- **generate**: a single structured-output call. The model decides, per turn:
  `intent` (clarify / recommend / refine / compare / refuse / finalize),
  `reply`, `recommended_ids` (the full current shortlist, not a delta), and
  `end_of_conversation`. One call instead of a multi-step agent pipeline is a
  deliberate choice: the assignment explicitly flags that a non-deterministic
  conversation shouldn't make the system fall apart, and every extra
  LLM-to-LLM handoff is another place for that to happen, plus another chunk
  of the 30s-per-call budget spent.
- **validate**: every id the model claims is looked up against the real
  catalog (`app/catalog.py::by_key`, matched by normalized name, URL, or
  entity_id). Anything that doesn't resolve is silently dropped, never
  returned. `end_of_conversation` is only ever trusted when `intent ==
  finalize` **and** the resolved shortlist is non-empty — a model claiming
  "done" with nothing to show doesn't get to end the conversation.

`recommendations` is empty on clarify/compare/refuse turns and the full
shortlist on recommend/refine/finalize — matching the reference conversations'
convention of only showing the table on turns that actually establish or
change it, not re-echoing it on every turn.

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
against the live agent in-process, building message history from the agent's
own real replies at each step (matching how a stateless caller actually
works). It reports Recall@10 on the final shortlist per trace, an agreement
rate on *whether* a table was shown per turn (soft signal — the reference is
one valid path, not the only one), and `end_of_conversation` agreement on the
final turn. This is scripted replay against gold user text, not a full
LLM-simulated user like the actual grading harness — faster to iterate with,
fully deterministic for regression-testing prompt changes, at the cost of not
testing how the agent handles a user who deviates from the reference script.
`tests/test_smoke.py` covers the plumbing (grounding validation, injection
short-circuit, turn cap, schema shape) with the LLM mocked out, so it runs
without an API key.

**TODO before submission:** run `python scripts/eval_harness.py` with a real
`GEMINI_API_KEY` and paste the actual mean Recall@10 and per-trace results
here, plus a short note on any prompt iteration that followed from a low
score on a specific trace (this section is intentionally left for the real
numbers rather than invented ones).

## AI tool usage disclosure

Built with Claude (Anthropic) as a pair-programming/design partner in an
agentic coding session: architecture discussion, the PDF-corruption
diagnosis and fix, and the initial implementation of every file in `app/`
and `scripts/` were done in that session. All design trade-offs above were
reviewed and are defensible on their own merits, not just "what the model
suggested."
