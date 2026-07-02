"""Local evaluation harness, run before ever deploying.

For each trace in traces/*.md: replay the user turns one at a time against
the live agent (in-process, not over HTTP -- faster iteration), building the
message history from OUR OWN agent's actual replies at each step, exactly as
a real stateless caller would. This is scripted replay against gold user
utterances, not a full LLM-simulated user like the real grader uses -- it's
faster to iterate with and fully deterministic for regression-testing changes
to the prompt/graph. See APPROACH.md for why that trade-off is fine here.

Reports:
  - Recall@10 on the FINAL shortlist per trace, averaged across traces
  - Recommendation-presence agreement: did our agent show a table exactly on
    the turns the reference did (soft signal -- the reference is one valid
    conversational path, not the only one)
  - end_of_conversation agreement on the final turn
  - Any turn that raised an exception or returned a malformed response

Requires GEMINI_API_KEY to be set (this exercises the real model).

Usage: python scripts/eval_harness.py
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import ChatMessage
from app.agent import run_turn
from scripts.trace_parser import GoldTurn, load_all_traces


def recall_at_k(expected: list[str], actual: list[str]) -> float | None:
    if not expected:
        return None
    exp = {u.rstrip("/") for u in expected}
    act = {u.rstrip("/") for u in actual}
    return len(exp & act) / len(exp)


def replay_trace(name: str, turns: list[GoldTurn]) -> dict:
    history: list[ChatMessage] = []
    final_recs: list[str] = []
    final_eoc = False
    rec_agreement = 0
    errors = []

    for i, turn in enumerate(turns, start=1):
        history.append(ChatMessage(role="user", content=turn.user_message))
        try:
            reply, recs, eoc = run_turn(history)
        except Exception as e:  # noqa: BLE001
            errors.append(f"turn {i}: {type(e).__name__}: {e}")
            break

        actual_urls = [r.url for r in recs]
        expected_has_table = bool(turn.expected_urls)
        actual_has_table = bool(actual_urls)
        if expected_has_table == actual_has_table:
            rec_agreement += 1

        history.append(ChatMessage(role="assistant", content=reply))
        final_recs = actual_urls
        final_eoc = eoc

    last_turn = turns[-1]
    recall = recall_at_k(last_turn.expected_urls, final_recs)

    return {
        "trace": name,
        "turns": len(turns),
        "recall_at_10": recall,
        "rec_presence_agreement": rec_agreement / len(turns) if turns else None,
        "eoc_match": final_eoc == last_turn.expected_end_of_conversation,
        "errors": errors,
    }


def main() -> None:
    traces = load_all_traces("traces")
    results = []
    for name, turns in traces.items():
        t0 = time.time()
        r = replay_trace(name, turns)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        status = "OK" if not r["errors"] else "ERROR"
        recall_str = f"{r['recall_at_10']:.2f}" if r["recall_at_10"] is not None else "n/a"
        print(
            f"[{status}] {name}: recall@10={recall_str}  "
            f"rec_presence_agreement={r['rec_presence_agreement']:.2f}  "
            f"eoc_match={r['eoc_match']}  ({r['seconds']}s)"
        )
        for err in r["errors"]:
            print(f"    ! {err}")

    recalls = [r["recall_at_10"] for r in results if r["recall_at_10"] is not None]
    print()
    print(f"Mean recall@10: {statistics.mean(recalls):.3f}" if recalls else "Mean recall@10: n/a")
    print(f"Traces with errors: {sum(1 for r in results if r['errors'])}/{len(results)}")


if __name__ == "__main__":
    main()
