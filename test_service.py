"""Tests for the discovery service's edges.

Run:  python3 llm_agents_call/test_service.py

Nothing here calls Gemini. What is worth testing without a key is everything
that decides whether a request is allowed to spend one, plus the parsing that
has to survive a model wrapping its JSON in a fence — those are the parts that
fail quietly, and the model call is the part that fails loudly.

Same hand-rolled runner as test_extract.py, so both run the same way with
nothing installed but the service's own dependencies.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set before the app imports, because require_token reads the environment on
# each call but healthz reports it at call time too.
os.environ.setdefault("DISCOVERY_TOKEN", "test-token-not-a-real-secret")

from fastapi.testclient import TestClient  # noqa: E402

import llm  # noqa: E402
import service  # noqa: E402

PASS = FAIL = 0
client = TestClient(service.app)


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")


print("\nHealth, which needs no token")
r = client.get("/healthz")
check("healthz is open", r.status_code, 200)
check("reports the model", r.json()["model"], llm.MODEL)
# A boolean, never the key or a mask of it.
check("key is a boolean", isinstance(r.json()["key_configured"], bool), True)
check("no key echoed", "gemini_api_key" in str(r.json()).lower(), False)


print("\nEvery other route needs the token")
check("no header is refused", client.get("/queries").status_code, 401)
check("wrong token is refused",
      client.get("/queries", headers={"X-Discovery-Token": "wrong"}).status_code, 401)

auth = {"X-Discovery-Token": os.environ["DISCOVERY_TOKEN"]}
check("right token is served", client.get("/queries", headers=auth).status_code, 200)
check("fifteen seed queries", len(client.get("/queries", headers=auth).json()["queries"]), 15)
# The year is computed, not written, so a query does not go stale in January.
from datetime import date  # noqa: E402
check("current year in the first query",
      str(date.today().year) in client.get("/queries", headers=auth).json()["queries"][0], True)

check("search refuses a body it cannot use",
      client.post("/search", headers=auth, json={"query": "hi"}).status_code, 422)


print("\nNo token configured means serve nothing, not serve everything")
saved = os.environ.pop("DISCOVERY_TOKEN")
check("unconfigured service refuses", client.get("/queries", headers=auth).status_code, 503)
os.environ["DISCOVERY_TOKEN"] = saved


print("\nSalvaging JSON the way a model actually returns it")
check("plain object", llm._as_json('{"urls": ["https://a.test"]}'), {"urls": ["https://a.test"]})
check("fenced", llm._as_json('```json\n{"ok": true}\n```'), {"ok": True})
check("fenced, unlabelled", llm._as_json('```\n{"ok": true}\n```'), {"ok": True})
check("prefaced with a sentence",
      llm._as_json('Here is what I found:\n{"ok": true}'), {"ok": True})
# The brace scanner has to balance, not stop at the first closer.
check("nested braces", llm._as_json('note\n{"a": {"b": 1}}'), {"a": {"b": 1}})
check("a bare array", llm._as_json('["https://a.test"]'), ["https://a.test"])

try:
    llm._as_json("I could not find anything useful.")
    check("prose raises", "no error", "ModelError")
except llm.ModelError:
    check("prose raises", "ModelError", "ModelError")


print("\nRules are refused, never coerced")
issues: list[str] = []
kept = llm._clean_rules([
    {"field": "disability_percent", "op": "GTE", "value": 40,
     "description": "This scheme needs a certified disability of 40% or more."},
    {"field": "annual_family_income", "op": "LTE", "value": 250000,
     "description": "This scheme is for families earning under Rs 2.5 lakh a year."},
    # Not a rule field.
    {"field": "favourite_colour", "op": "EQ", "value": "blue", "description": "x" * 20},
    # A real field, a value no profile holds.
    {"field": "course_level", "op": "IN", "value": ["UNDERGRADUATE", "B.Tech"],
     "description": "This scheme is for undergraduates."},
    # Out of range.
    {"field": "disability_percent", "op": "GTE", "value": 140, "description": "x" * 20},
], issues)

check("two rules kept", [r["field"] for r in kept],
      ["disability_percent", "annual_family_income", "course_level"])
check("unknown field dropped",
      any("favourite_colour" in i for i in issues), True)
check("unknown choice value dropped, field kept",
      [r["value"] for r in kept if r["field"] == "course_level"], [["UNDERGRADUATE"]])
check("the bad value is named", any("B.Tech" in i for i in issues), True)
check("second rule on a field dropped",
      sum(1 for r in kept if r["field"] == "disability_percent"), 1)
check("every kept rule is hard", {r["hard"] for r in kept}, {True})

# A rule with no usable sentence still has to carry one, because the API
# requires it and a refused student is shown it.
generated: list[str] = []
made = llm._clean_rules(
    [{"field": "disability_percent", "op": "GTE", "value": 75, "description": "too short"}],
    generated)
check("short sentence is replaced", made[0]["description"],
      "This scheme needs a certified disability of 75% or more.")
check("and the operator is told", any("generated from the rule" in i for i in generated), True)


print("\nA notice with no stated threshold gets no threshold")
none: list[str] = []
check("empty in, empty out", llm._clean_rules([], none), [])
check("nothing invented", none, [])


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
