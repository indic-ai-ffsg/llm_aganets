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
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set before the app imports, because require_token reads the environment on
# each call but healthz reports it at call time too.
os.environ.setdefault("DISCOVERY_TOKEN", "test-token-not-a-real-secret")

from fastapi.testclient import TestClient  # noqa: E402

import crosscheck  # noqa: E402
import llm  # noqa: E402
import service  # noqa: E402
import vocab  # noqa: E402

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


print("\nThe page is fetched here, not browsed by the model")
# Where the thirty-to-sixty seconds went. A call carrying url_context is the
# model deciding to browse, fetching, and waiting on a government web server
# before it starts reading; handed the text it is one ordinary turn.
_asked = {}
_real_ask = llm._ask


def _spy(prompt, tool, api_key):
    _asked['tool'] = tool
    _asked['prompt'] = prompt
    return ('{"title": "Asha", "sponsor_name": "SBI Foundation", '
            '"summary": "a summary comfortably over twenty characters", "rules": []}')


llm._ask = _spy
try:
    llm.extract("https://x.test/scheme", ["Engineering"],
                page_text="Family income not exceeding Rs. 2.50 lakh per annum.")
    check("a page in hand means no browse tool", _asked['tool'], None)
    check("and the page is in the prompt",
          "----- the page -----" in _asked['prompt'], True)
    check("and it still says where the page came from",
          "https://x.test/scheme" in _asked['prompt'], True)

    llm.extract("https://x.test/scheme", [])
    check("no page means the model browses, as before", _asked['tool'], "url_context")
finally:
    llm._ask = _real_ask


print("\nA page that builds itself in the browser is not a page")
# The case that would otherwise ship a draft made of a page title. An HTTP GET
# against a single-page app succeeds and returns a shell — the real
# www.sbiashascholarship.co.in returns 351 characters, of which the content is
# the <title>. Handing that to a model produces a confident answer from memory.
check("a JavaScript shell is thin",
      llm.too_thin("SBI Asha Scholarship 2026-27 | SBI Foundation\n\n\n" + " \n" * 90),
      True)
check("nothing fetched is thin", llm.too_thin(""), True)
check("and so is nothing at all", llm.too_thin(None), True)
check("a real notice is not",
      llm.too_thin("Eligibility. " * 200), False)


print("\nBoth readers are given the same bytes")
# Two fetches can return two different pages — a rotating banner, a notice
# edited between them — and a disagreement caused by that is a false alarm an
# operator cannot tell from a real one.
_fetched = []
_real_fetch = crosscheck.fetch_text


def _no_fetch(url, timeout=20.0):
    _fetched.append(url)
    raise AssertionError("fetched again when the text was already in hand")


crosscheck.fetch_text = _no_fetch
try:
    chk = crosscheck.against("https://x.test/scheme", [], None,
                             text="Last date to apply: 31 October 2026")
    check("the supplied text is used", chk.ran, True)
    check("nothing was fetched a second time", _fetched, [])
    check("and it was read", chk.closes_at, "2026-10-31")
finally:
    crosscheck.fetch_text = _real_fetch


print("\nThe comparison is the field's, not the model's")
# The failure this section exists for: `annual_family_income GTE 10000` is a
# well-formed rule the API accepts and the matcher evaluates, and it inverts who
# the scheme is for. Nothing downstream reads as wrong, which is why it has to be
# caught here.
reversed_op: list[str] = []
flipped = llm._clean_rules([{
    "field": "annual_family_income", "op": "GTE", "value": 250000,
    "description": "This scheme is for families earning Rs 2.5 lakh a year or more.",
}], reversed_op)

check("a reversed comparison is corrected, not accepted",
      [r["op"] for r in flipped], ["LTE"])
check("the rule survives it", [r["value"] for r in flipped], [250000])
check("and the operator is told what the model said",
      any("proposed GTE" in i for i in reversed_op), True)
check("and told to re-check the figure",
      any("wrong sentence" in i for i in reversed_op), True)
# The sentence went with it, because it is the one a refused student reads and
# it was written to describe the reversed rule.
check("the contradicting refusal sentence is replaced",
      flipped[0]["description"],
      "This scheme is for families with an annual income of \u20b92,50,000 or less.")
check("and that replacement is reported too",
      any("said the opposite of the rule" in i for i in reversed_op), True)

# Indian grouping, because a refused student reads this sentence and the whole
# platform pins en-IN for that reason (admin/src/lib/format.ts).
check("a lakh figure groups the Indian way", llm._rupees(250000), "2,50,000")
check("and so does a crore", llm._rupees(15000000), "1,50,00,000")
check("under a thousand is left alone", llm._rupees(900), "900")

# The floor field, the other way round.
floored: list[str] = []
llm._clean_rules([{
    "field": "disability_percent", "op": "LTE", "value": 40,
    "description": "This scheme is for students with up to 40% disability.",
}], floored)
check("a disability percentage is a floor, however it arrives",
      any("proposed LTE" in i for i in floored), True)

# The ordinary case has to stay silent, or the panel fills with notes nobody
# reads and the real ones are lost among them.
quiet: list[str] = []
right = llm._clean_rules([{
    "field": "annual_family_income", "op": "LTE", "value": 250000,
    "description": "This scheme is for families earning under Rs 2.5 lakh a year.",
}], quiet)
check("the right comparison passes", [r["op"] for r in right], ["LTE"])
check("and says nothing", quiet, [])
check("and keeps the notice's own wording", right[0]["description"],
      "This scheme is for families earning under Rs 2.5 lakh a year.")

# The model is no longer asked for an operator at all, so most rules arrive
# without one.
absent: list[str] = []
filled = llm._clean_rules([{
    "field": "course_level", "value": ["UNDERGRADUATE"],
    "description": "This scheme is for undergraduate students.",
}], absent)
check("a rule with no operator gets the field's", [r["op"] for r in filled], ["IN"])
check("and that is not worth a note", absent, [])


print("\nThe vocabulary tables agree with each other")
# All three are import-time assertions in vocab.py; asserting them here as well
# is what makes a failure say which one, in a suite somebody runs, rather than
# only killing the container.
check("every proposable field has a meaning for the prompt",
      set(vocab.PROPOSABLE_RULES) ^ set(vocab.RULE_MEANING), set())
check("every declared operator is one the API accepts",
      {op for op in vocab.PROPOSABLE_RULES.values()} - vocab.RULE_OPS, set())
check("every set-valued field has a domain to check against",
      {f for f, op in vocab.PROPOSABLE_RULES.items()
       if op in ("IN", "NOT_IN") and f not in vocab.CHOICE_DOMAINS}, set())

# enums.py is generated and committed, and has now fallen behind the migrations
# twice. The second time nothing here could start at all — vocab asked for a
# name enums did not define. This is the message that failure produces now.
try:
    import enums as _enums
    _held = _enums.SOCIAL_CATEGORIES
    del _enums.SOCIAL_CATEGORIES
    try:
        vocab._generated("SOCIAL_CATEGORIES")
        check("a stale enums.py names the fix", "no error", "ImportError")
    except ImportError as e:
        check("a stale enums.py names the fix", "generate_enums.py" in str(e), True)
finally:
    _enums.SOCIAL_CATEGORIES = _held


print("\nA notice with no stated threshold gets no threshold")
none: list[str] = []
check("empty in, empty out", llm._clean_rules([], none), [])
check("nothing invented", none, [])


print("\nA rule that permits everything is not a rule")
# The engine tests presence before it evaluates, so an all-inclusive rule still
# blocks a student who has not filled the field in — and whatever they answer,
# they pass. That is the "add your gender to your profile" prompt that cannot
# change any outcome.
everything: list[str] = []
allfour = llm._clean_rules([{
    "field": "gender", "op": "IN",
    "value": ["MALE", "FEMALE", "TRANSGENDER", "UNDISCLOSED"],
    "description": "This scheme is open to applicants of any gender.",
}], everything)
check("all four genders is dropped", allfour, [])
check("and the reason says why", any("restricts nobody" in i for i in everything), True)

# The narrowing case still has to survive, or a women-only scheme stops being
# women-only — which is the opposite failure and a worse one.
somegenders: list[str] = []
kept_one = llm._clean_rules([{
    "field": "gender", "op": "IN", "value": ["FEMALE"],
    "description": "This scheme is for women applicants only.",
}], somegenders)
check("a genuine restriction is kept", [r["value"] for r in kept_one], [["FEMALE"]])
check("and nothing is said about it", somegenders, [])

# Three of four is still a restriction, so it stays.
partial: list[str] = []
kept_three = llm._clean_rules([{
    "field": "gender", "op": "IN",
    "value": ["MALE", "FEMALE", "TRANSGENDER"],
    "description": "This scheme is open to all but undisclosed applicants.",
}], partial)
check("a subset is kept", len(kept_three), 1)


print("\nDates the model wrote, read or refused")
check("ISO", llm._parse_date("2026-10-31"), "2026-10-31")
check("day first, as an Indian notice writes it",
      llm._parse_date("31/10/2026"), "2026-10-31")
check("dotted", llm._parse_date("31.10.2026"), "2026-10-31")
check("named month", llm._parse_date("31 October 2026"), "2026-10-31")
check("month first", llm._parse_date("October 31, 2026"), "2026-10-31")
# The shape check this replaced accepted these, and the Go side then stored no
# date at all — so an impossible date and an absent one were the same thing.
check("a month that does not exist", llm._parse_date("2026-13-01"), None)
check("a day that does not exist", llm._parse_date("2026-02-31"), None)
check("prose", llm._parse_date("some time in the autumn"), None)
check("nothing", llm._parse_date(None), None)
check("empty", llm._parse_date("   "), None)


print("\nWhat is said about a proposal's dates")
TODAY = date(2026, 9, 7)


def issues_for(**kw):
    return llm._date_issues(llm.Proposal(**kw), TODAY)


check("a deadline that has passed is named",
      any("has already passed" in i for i in issues_for(closes_at="2024-10-31")), True)
check("and the reason is the one that matters",
      any("removes the scheme from the directory" in i
          for i in issues_for(closes_at="2024-10-31")), True)
check("no deadline is said, not complained about",
      any("no closing date was read" in i for i in issues_for()), True)
check("a live deadline says nothing",
      issues_for(closes_at="2026-10-31"), [])
check("an opening date years out is named",
      any("more than eighteen months away" in i
          for i in issues_for(closes_at="2026-10-31", opens_at="2029-01-01")), True)


print("\nSettling the deadline between the two readers")


def settled(model_date, pattern_date, inferred=False, opens_at=None):
    p = llm.Proposal(title="x", closes_at=model_date, opens_at=opens_at)
    c = crosscheck.Check(closes_at=pattern_date, closes_at_inferred=inferred)
    service.settle_closing_date(p, c, TODAY)
    return p


check("the page's date is adopted when the model gave none",
      settled(None, "2026-10-31").closes_at, "2026-10-31")
check("and the operator is told where it came from",
      any("read off the page" in i for i in settled(None, "2026-10-31").issues), True)
check("an inferred year is called out",
      any("no year" in i for i in settled(None, "2026-10-31", inferred=True).issues), True)

# The case this exists for: the model keeps the day and month and supplies a
# year from whenever it last saw the page.
check("a stale year loses to an evidenced live one",
      settled("2024-10-31", "2026-10-31").closes_at, "2026-10-31")
check("and the consequence is spelled out",
      any("out of the directory" in i for i in settled("2024-10-31", "2026-10-31").issues), True)

# Everything else is left alone: which reader is right is the operator's call.
check("two future dates that differ are not settled here",
      settled("2026-11-30", "2026-10-31").closes_at, "2026-11-30")
check("and nothing is added to the issues",
      settled("2026-11-30", "2026-10-31").issues, [])
check("agreement changes nothing", settled("2026-10-31", "2026-10-31").issues, [])
check("no pattern date, nothing to settle",
      settled("2024-10-31", None).closes_at, "2024-10-31")

# The adopted date can cross an opening date the model gave, and the API
# refuses that pair outright.
crossed = settled(None, "2026-10-31", opens_at="2026-12-01")
check("an opening date past the adopted deadline is dropped", crossed.opens_at, None)
check("and said", any("was dropped" in i for i in crossed.issues), True)


print("\nTranslation asks for languages it can actually name")
got, issues = llm.translate({"summary": "x"}, ["xx", "klingon"])
check("an unknown code is refused, not guessed at", got, {})
check("and named", any("'xx'" in i for i in issues), True)
# English is the original. Asking for it is not an error and not a call.
check("english is not translated into", llm.translate({"summary": "x"}, ["en"]), ({}, []))
check("nothing to translate is not a call", llm.translate({}, ["hi"]), ({}, []))
check("bengali is a language this will translate into", "bn" in llm.LANGUAGE_NAMES, True)


print("\nWhat comes back from a translation, kept or reported")
english = {"summary": "A scholarship for students with disabilities.",
           "description": "The full description."}
notes: list[str] = []
kept = llm._take_translations(
    {"hi": {"summary": "दिव्यांग विद्यार्थियों के लिए छात्रवृत्ति।",
            "description": "पूरा विवरण।"}},
    ["hi"], english, {}, notes)
check("a good reply is kept", sorted(kept["hi"]), ["description", "summary"])
check("and says nothing", notes, [])

notes = []
kept = llm._take_translations(
    {"hi": {"summary": english["summary"], "description": "पूरा विवरण।"}},
    ["hi"], english, {}, notes)
check("text echoed back untranslated is kept but flagged",
      any("identical to the English" in i for i in notes), True)
check("and still stored, because a proper noun legitimately is",
      kept["hi"]["summary"], english["summary"])

notes = []
kept = llm._take_translations(
    {"hi": {"summary": "क" * 900, "description": "पूरा विवरण।"}},
    ["hi"], english, {"summary": 600}, notes)
check("a translation longer than its column is cut, not refused",
      len(kept["hi"]["summary"]), 600)
check("and the cut is reported", any("was cut" in i for i in notes), True)

notes = []
kept = llm._take_translations({"hi": "not an object"}, ["hi"], english, {}, notes)
check("a language that came back unusable is dropped", kept, {})
check("and named", any("nothing usable" in i for i in notes), True)

notes = []
kept = llm._take_translations(
    {"hi": {"summary": "दिव्यांग विद्यार्थियों के लिए छात्रवृत्ति।"}},
    ["hi"], english, {}, notes)
check("a partial reply keeps what it has", list(kept["hi"]), ["summary"])
check("and says what is missing",
      any("no translation came back for description" in i for i in notes), True)


print("\nThe translate route")
check("needs the token too", client.post("/translate", json={}).status_code, 401)
r = client.post("/translate", headers=auth,
                json={"fields": {"summary": "x"}, "languages": ["xx"]})
check("answers without a key when there is no call to make", r.status_code, 200)
check("and reports the refused language", r.json()["translations"], {})


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
