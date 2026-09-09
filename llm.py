"""The three model-backed stages: find pages, judge them, read them.

What changed from the keyless version, and what did not
------------------------------------------------------
extract.py's argument still stands and it is still running — see crosscheck.py.
What a model adds is the stage that never existed here: *finding* a notice.
`pipeline.py --url` only ever read a page somebody had already found, so
"discover new scholarships" was never something this could do.

So the division is deliberate:

    search    a model with a search tool. Nothing else can do this.
    classify  a model. "Is this a real, open, education scholarship for
              disabled students in India" is a judgement, and the keyless
              pipeline explicitly refused to make it.
    extract   a model, then checked against extract.py's regexes on the same
              page. The model reads prose the patterns cannot; the patterns
              carry evidence the model cannot be trusted to. Where the two
              disagree on a number that decides eligibility, the operator is
              told, and nothing is imported on the model's word alone.

Ported from the Next.js portal's src/lib/llm.ts, which was itself ported from
llm_agentic_call.py. The prompts survive; the output shape does not, because
this platform's schema is not that one — a listing here carries eligibility
*rules* the matcher evaluates, not a flat min_disability_percentage column.

The SDK call is models.generate_content rather than the TS port's
interactions.create: the interactions surface is not in the public Python SDK,
and the background/polling mode the old code wanted was disabled anyway (its
key got 403 on GET /interactions). These calls are synchronous, ~30-60s each,
which is why the backend runs them one query at a time on a worker.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enums import COURSE_LEVELS, DISABILITY_TYPES  # noqa: E402
# The month table, borrowed rather than copied. A model asked for YYYY-MM-DD
# sometimes answers "31 October 2026" instead, and a third spelling of the
# twelve months is a third one to keep in step.
from extract import MONTHS  # noqa: E402
from vocab import (  # noqa: E402
    AWARD_BASES, CHOICE_DOMAINS, PROPOSABLE_RULES, RULE_MEANING, SPONSOR_TYPES,
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

# How many search hits are worth classifying per query. Each URL costs a
# classify call and possibly an extract call, both ~30-60s, so this number is
# the main lever on how long a run takes.
MAX_URLS_PER_QUERY = int(os.environ.get("DISCOVERY_MAX_URLS_PER_QUERY", "10"))


class ModelError(RuntimeError):
    """The model could not be reached, or answered something unusable."""


def _client(api_key: str | None = None):
    """A Gemini client.

    The key may arrive per request so an operator can supply their own when the
    service's key expires or hits quota — the same escape hatch the old admin
    panel had, kept because it is the one that gets used at 11pm.
    """
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise ModelError(
            "No Gemini API key. Set GEMINI_API_KEY on the service, or pass one "
            "with the request.")
    try:
        from google import genai
    except ImportError as e:  # pragma: no cover - depends on the image
        raise ModelError(
            "google-genai is not installed in this image; see "
            "requirements-service.txt") from e
    return genai.Client(api_key=key)


def _tool(kind: str):
    """A tool declaration, by the name the SDK gives it.

    google_search is what finds pages; url_context is what lets the model read
    one it has not been handed the text of. Both are declared through types.Tool
    rather than a bare dict so a rename in the SDK fails here, loudly, instead
    of being silently ignored and leaving a model that answers from memory.
    """
    from google.genai import types

    if kind == "google_search":
        return types.Tool(google_search=types.GoogleSearch())
    if kind == "url_context":
        return types.Tool(url_context=types.UrlContext())
    raise ValueError(f"unknown tool {kind!r}")


def _ask(prompt: str, tool: str | None, api_key: str | None) -> str:
    """One turn, with at most one tool, returning raw text.

    A tool and a JSON response type cannot both be set on the same call, so the
    shape is asked for in the prompt and parsed defensively by _as_json. Where
    there is no tool — translate, and the fast read that is handed the page —
    the JSON mime type is set as well: it costs nothing, and it takes the fences
    and the leading "Here is the JSON you asked for" out of the reply rather
    than out of the parser. _as_json stays, because it is what catches the times
    the mime type is ignored.
    """
    from google.genai import types

    client = _client(api_key)
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[_tool(tool)] if tool else None,
                response_mime_type=None if tool else "application/json",
                # Deterministic on purpose. Two runs over the same notice
                # should propose the same income ceiling, and a reviewer who
                # re-reads a proposal should see what they saw before.
                temperature=0.0,
            ),
        )
    except Exception as e:  # the SDK raises a family of transport errors
        raise ModelError(f"{MODEL} call failed: {e}") from e

    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise ModelError(f"{MODEL} returned nothing")
    return text


def _as_json(text: str) -> dict | list:
    """The first JSON value in the reply.

    A model asked for JSON alongside a search tool very often wraps it in a
    fence or prefaces it with a sentence. Salvaging that is not sloppiness — it
    is cheaper than a failed run, and a reply that genuinely is not JSON still
    raises rather than being guessed at.
    """
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # A leading sentence, then the object. Braces are matched by scanning
    # rather than by regex, which cannot balance them.
    start = min((i for i in (body.find("{"), body.find("[")) if i != -1), default=-1)
    if start != -1:
        depth = 0
        opener = body[start]
        closer = "}" if opener == "{" else "]"
        for i in range(start, len(body)):
            if body[i] == opener:
                depth += 1
            elif body[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(body[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ModelError(f"reply was not JSON: {text[:200]}")


# --- dates -------------------------------------------------------------------
#
# Dates get their own section because they are the field where the model's
# failure is both characteristic and invisible. Asked when a scheme closes, a
# model that has seen the page before will answer with the right day and month
# and a year from whenever it saw it — confidently, in the requested format,
# indistinguishable from a reading.
#
# What that costs is specific to this platform. The public directory selects
#
#     (closes_at IS NULL OR closes_at > now())
#     AND (opens_at IS NULL OR opens_at <= now())
#
# (migration 0043), so a stale year does not show a student "this closed". It
# takes the scheme off the site, and the panel goes on saying "Published". Not
# knowing a date is safe here; guessing one is not, and every rule below follows
# from that asymmetry.

# Roughly eighteen months. A scholarship cycle is annual, so a window wider than
# this is a year gone astray far more often than it is a real announcement.
FAR_AHEAD_DAYS = 550


def _iso(y: int, mon: int, d: int) -> str | None:
    try:
        return date(y, mon, d).isoformat()
    except ValueError:
        return None


def _parse_date(raw) -> str | None:
    r"""A date the model wrote, as YYYY-MM-DD, or None.

    The prompt asks for ISO and usually gets it. The other three forms are what
    comes back when the model has copied the notice's own wording, and reading
    them is cheaper than dropping a date the page really does state.

    Anything else is None rather than a guess. This replaces a bare
    `re.fullmatch(r"\d{4}-\d{2}-\d{2}")`, which accepted 2026-13-45 — the Go
    side then parsed it, failed, and stored no date at all, so an impossible
    date and an absent one were the same thing and neither was reported.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None

    months = "|".join(sorted(MONTHS, key=len, reverse=True))

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return _iso(int(m[1]), int(m[2]), int(m[3]))

    # Day first, which is how an Indian notice writes it and therefore how a
    # model copying one writes it back.
    m = re.fullmatch(r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})", s)
    if m:
        return _iso(int(m[3]), int(m[2]), int(m[1]))

    m = re.fullmatch(
        r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+(?:of\s+)?(" + months + r")[a-z]*\.?,?\s+(\d{4})",
        s, re.I)
    if m:
        return _iso(int(m[3]), MONTHS[m[2].lower()[:3]], int(m[1]))

    m = re.fullmatch(
        r"(" + months + r")[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s, re.I)
    if m:
        return _iso(int(m[3]), MONTHS[m[1].lower()[:3]], int(m[2]))

    return None


def _date_issues(p: "Proposal", today: date | None = None) -> list[str]:
    """What is suspicious about a proposal's dates, said rather than corrected.

    Nothing here changes a value, because each of these is a thing a notice can
    honestly say and the proposal is going to a person either way. They are
    worth saying because none of them looks like a problem in the panel: a
    listing with a stale closes_at is published, shows as Published, and is
    absent from the directory, with nothing anywhere connecting the three.

    The missing-deadline line is not a complaint. A listing with no closes_at is
    publishable and is shown as open — that is what 0043 decided — and the note
    exists so an operator knows that is what they are publishing.
    """
    today = today or date.today()
    horizon = (today + timedelta(days=FAR_AHEAD_DAYS)).isoformat()
    out: list[str] = []

    if p.closes_at and p.closes_at < today.isoformat():
        out.append(
            f"closes_at {p.closes_at} has already passed, although the page was "
            "judged open. Check the year — a closing date in the past removes the "
            "scheme from the directory rather than showing it as closed.")
    elif not p.closes_at:
        out.append(
            "no closing date was read. The listing can still be published and will "
            "show as open, but a student is told nothing about when to apply by.")

    if p.opens_at and p.opens_at > horizon:
        out.append(
            f"opens_at {p.opens_at} is more than eighteen months away, which would "
            "keep the listing hidden until then. Check the year.")
    if p.closes_at and p.closes_at > horizon:
        out.append(
            f"closes_at {p.closes_at} is more than eighteen months away, which is "
            "longer than an annual cycle. Check the year.")

    return out


# --- stage 1: search ---------------------------------------------------------

def search(query: str, api_key: str | None = None) -> list[str]:
    """Candidate URLs for one query.

    The prompt's exclusions are the ones that cost review time: an aggregator
    listing twenty schemes produces one draft naming none of them, and a news
    article about a scheme is not the scheme.
    """
    prompt = f"""Search for: {query}

Today is {date.today().isoformat()}.

Find URLs of specific scholarship programmes available to students with
disabilities in India.

Prefer the page for the cycle that is open or next to open. Where a scheme has
a page per year, the current one is wanted and an archived earlier one is not.

These must be education scholarships (school, college, university, or
vocational) — not employment schemes, pensions, assistive device grants, or
general welfare programmes.

Return only direct canonical scholarship programme pages — not news articles,
not aggregator listings, not blog posts.

Return up to {MAX_URLS_PER_QUERY} URLs as JSON:
{{"urls": ["https://...", "https://..."]}}"""

    data = _as_json(_ask(prompt, "google_search", api_key))
    urls = data.get("urls") if isinstance(data, dict) else data
    if not isinstance(urls, list):
        raise ModelError("search did not return a list of urls")

    out: list[str] = []
    for u in urls:
        if isinstance(u, str) and u.startswith(("http://", "https://")) and u not in out:
            out.append(u)
    return out[:MAX_URLS_PER_QUERY]


# --- stage 2: classify -------------------------------------------------------

@dataclass
class Verdict:
    is_scholarship: bool
    is_open: bool
    reason: str = ""
    # The deadline the model says it judged on, where the page gave one.
    #
    # Not forwarded to the registry — extract reads the page again for that —
    # and the Go client does not decode it. It is here to make `is_open`
    # checkable: asked to commit to a date, a model is answerable for it, and
    # the check below is the only part of this stage that is arithmetic rather
    # than judgement.
    closes_at: str | None = None


def classify(url: str, api_key: str | None = None) -> Verdict:
    """Whether this page is a live education scholarship for disabled students.

    The judgement the keyless pipeline refused to make, and the reason it
    refused still applies — so the answer is not final. A page that passes here
    becomes a DRAFT for an operator to approve, never a published listing.
    """
    prompt = f"""Analyse this URL: {url}

Today is {date.today().isoformat()}.

1. Is this a scholarship, fellowship or grant for EDUCATION, available to
   students with disabilities in India?
   - It must fund education: school, college, university, or vocational training.
   - It must be open to students with disabilities as defined by the RPWD Act
     2016 — either disability-specific, or open to all including them.
   - It must be for students in India.
   - Reject an employment or job scheme, a pension, an assistive-device grant, a
     general welfare programme, a news article, a blog post, or an aggregator
     page listing several schemes.
   - Disease-support programmes and schemes for other vulnerable groups are not
     disability scholarships.

2. Is it currently open or upcoming?
   - Judge this against today's date above and against what the page says now,
     never against a cycle you remember. A page you have seen before will have
     moved on.
   - A deadline clearly passed with no sign of a new cycle: not open.
   - A recurring annual scholarship with no dates stated: treat as open.
   - An annual scheme between cycles, whose page is still maintained: open. A
     page rejected here is recorded and never read again, so being wrong in this
     direction loses the scheme permanently.
   - Discontinued: not open.

Give `closes_at` as the deadline the page states, in YYYY-MM-DD, so the answer
can be checked. Use null if the page states none — do not supply one from
memory.

Answer as JSON:
{{"is_scholarship": true, "is_open": true, "closes_at": "YYYY-MM-DD or null",
  "reason": "short reason if either is false"}}"""

    data = _as_json(_ask(prompt, "url_context", api_key))
    if not isinstance(data, dict):
        raise ModelError("classify did not return an object")

    v = Verdict(
        is_scholarship=bool(data.get("is_scholarship")),
        is_open=bool(data.get("is_open")),
        reason=str(data.get("reason") or "")[:300],
        closes_at=_parse_date(data.get("closes_at")),
    )

    # The arithmetic corrects the judgement, and only in the direction that
    # cannot lose a scheme.
    #
    # The asymmetry is in the backend, not here. A URL judged not-open is
    # written into discovery_seen_url with its reason, and no later run reads it
    # again — so a wrong "closed" is permanent, while a wrong "open" costs one
    # draft that an operator declines in a second. A model that says closed
    # while quoting a deadline that has not arrived is therefore overruled; one
    # that says open is left alone, and extract's date checks pick up the rest.
    if v.is_scholarship and not v.is_open and v.closes_at:
        if v.closes_at >= date.today().isoformat():
            v.is_open = True
            v.reason = (
                f"judged closed, but the deadline it quotes ({v.closes_at}) has not "
                f"passed, so it is being read as open. Original reason: "
                f"{v.reason or 'none given'}")[:300]

    return v


# --- stage 3: extract --------------------------------------------------------

@dataclass
class Proposal:
    """A curated listing as the API would take it, plus what to review.

    The field names are registry.CuratedInput's JSON names on purpose: the
    backend forwards this almost verbatim, so a name invented here would be
    dropped silently by the binder.
    """
    title: str = ""
    sponsor_name: str = ""
    sponsor_type: str = ""
    summary: str = ""
    description: str = ""
    external_url: str = ""
    academic_year: str = ""
    award_basis: str = ""
    award_amount: float | None = None
    opens_at: str | None = None
    closes_at: str | None = None
    benefit_summary: str = ""
    benefit_description: str = ""
    eligibility_summary: str = ""
    application_process: str = ""
    documents_required: list[str] = field(default_factory=list)
    important_notes: str = ""
    contact_email: str = ""
    contact_phone: str = ""
    tags: list[str] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)

    # Not sent to the API. What the operator has to look at before publishing.
    issues: list[str] = field(default_factory=list)


def _rupees(value) -> str:
    """A rupee figure in Indian digit grouping: 250000 -> 2,50,000.

    Not f"{n:,}", which was what this used and which writes 250,000.

    The sentence this ends up in is what a student who does not qualify is shown
    in place of the scheme, so it is read by the person the platform most owes a
    clear answer to. "2,50,000" reads as two and a half lakh at a glance; a
    reader who grew up with this system has to count the digits of "250,000".

    The panel pins Intl to en-IN throughout for the same reason, in the same
    words — see admin/src/lib/format.ts — so a sentence generated here and one
    typed by an operator now look alike. Written out rather than taken from
    `locale`, which needs en_IN installed in the image and silently falls back
    to the C locale when it is not: the wrong grouping would come back on the
    server and nowhere a test could see it.
    """
    n = int(value)
    digits = str(abs(n))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return ("-" if n < 0 else "") + digits


def _rule_sentence(field_name: str, op: str, value) -> str:
    """What a blocked student is told, when the model did not say it.

    RuleInput.description is required and 10-240 characters, and it is shown to
    the refused applicant word for word. A proposal without one would be
    refused by the API, so a plain sentence is generated rather than losing the
    scheme — and the operator is told it was generated, because "generated from
    the rule" is not the same as "checked against the notice".
    """
    if field_name == "disability_percent":
        return f"This scheme needs a certified disability of {value}% or more."
    if field_name == "annual_family_income":
        return (f"This scheme is for families with an annual income of "
                f"₹{_rupees(value)} or less.")
    if field_name == "course_level":
        levels = ", ".join(str(v).replace("_", " ").lower() for v in _as_list(value))
        return f"This scheme is for students at these levels of study: {levels}."
    if field_name == "state_code":
        return ("This scheme is limited to students domiciled in "
                f"{', '.join(_as_list(value))}.")
    if field_name == "disability_type":
        kinds = ", ".join(str(v).replace("_", " ").lower() for v in _as_list(value))
        return f"This scheme is for students with these conditions: {kinds}."
    if field_name == "gender":
        who = ", ".join(str(v).lower() for v in _as_list(value))
        return f"This scheme is open to {who} applicants only."
    if field_name == "social_category":
        cats = ", ".join(str(v) for v in _as_list(value))
        return f"This scheme is for applicants in these categories: {cats}."
    return f"This scheme requires {field_name.replace('_', ' ')} {op} {value}."


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _clean_rules(raw, issues: list[str]) -> list[dict]:
    """Keep the rules the matcher can actually evaluate; report the rest.

    Nothing is coerced. A rule naming a field or a value that is not in the
    vocabulary is dropped and named in `issues`, because the alternative —
    fuzzy-matching it into something plausible — is how a scheme ends up with
    an eligibility condition nobody wrote.
    """
    out: list[dict] = []
    if not isinstance(raw, list):
        return out

    for r in raw:
        if not isinstance(r, dict):
            continue
        name = str(r.get("field") or "").strip()
        if name not in PROPOSABLE_RULES:
            if name:
                issues.append(f"rule dropped: {name!r} is not a rule field")
            continue

        # The comparison is a property of the field, not an answer.
        #
        # This used to take the model's operator and check it against RULE_OPS —
        # every operator the API accepts, ten of them — so
        # `annual_family_income GTE 10000` passed. GTE is a real operator, and
        # that no income condition on any scholarship is a floor was checked by
        # nothing.
        #
        # What that costs is the worst failure in this pipeline, because no part
        # of it looks wrong. The rule is well-formed, the API stores it, the
        # matcher evaluates it, the panel renders it — and the scheme meant for
        # families under Rs 2.5 lakh now matches families above it and turns away
        # every student it was written for. No error is raised and no row is
        # missing.
        #
        # So the operator is not read. PROPOSABLE_RULES already declares the one
        # each field takes: a certificate percentage is always a floor, an income
        # figure always a ceiling, a set of levels always membership.
        #
        # That is not the coercion this function refuses everywhere else.
        # Coercing a *value* means guessing what a notice said; this is refusing
        # to let the model overwrite a constant the prompt handed it.
        #
        # The disagreement is still reported, and not only for the operator's
        # sake — a reading that reversed the comparison was not a careful one,
        # and the figure attached to it is worth a second look.
        op = PROPOSABLE_RULES[name]
        proposed = str(r.get("op") or op).strip().upper()
        corrected = proposed != op
        if corrected:
            issues.append(
                f"rule {name}: the model proposed {proposed}, which is not the "
                f"comparison this field takes — {op} was used instead. Check the "
                "figure as well: a reading that reversed the comparison may have "
                "taken the number from the wrong sentence.")

        value = r.get("value")
        if value is None or value == [] or value == "":
            issues.append(f"rule dropped: {name} has no value")
            continue

        if name in CHOICE_DOMAINS:
            allowed = CHOICE_DOMAINS[name]
            kept = [v for v in _as_list(value) if v in allowed]
            rejected = [v for v in _as_list(value) if v not in allowed]
            if rejected:
                issues.append(
                    f"rule {name}: dropped {', '.join(map(str, rejected))} — "
                    "not a stored value")
            if not kept:
                continue

            # A choice rule that permits every value restricts nobody, and
            # proposing one is worse than proposing none.
            #
            # The engine checks whether the profile HOLDS the field before it
            # evaluates the rule (matching/engine.go: `if !present ||
            # value.IsNull()` comes first), so an all-inclusive gender rule
            # still blocks every student who has not stated a gender — and
            # whatever they then answer, they pass. The student is sent to fill
            # in a field that cannot change the answer.
            #
            # scholarship-vocabulary.ts warns about this in the panel, in these
            # words: "Leave every box clear for a scheme open to everyone —
            # that is what 'no restriction' means to the engine. Ticking all
            # four is not the same thing and makes the matcher do work for
            # nothing." A model listing every category it saw named on the page
            # is exactly how that gets ticked.
            if set(kept) >= set(allowed):
                issues.append(
                    f"rule {name}: dropped — it listed every possible value, so it "
                    "restricts nobody, and a rule like that still blocks anyone who "
                    "has not filled the field in")
                continue

            value = kept
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                issues.append(f"rule dropped: {name} value {value!r} is not a number")
                continue
            if name == "disability_percent" and not 0 <= value <= 100:
                issues.append(f"rule dropped: disability_percent {value} outside 0-100")
                continue
            if name == "annual_family_income" and value < 0:
                issues.append("rule dropped: annual_family_income is negative")
                continue
            if value == int(value):
                value = int(value)

        # A sentence written to match a comparison that has just been corrected
        # describes the opposite rule, and it is the sentence a blocked student
        # is shown in place of the scheme. Left alone, a ceiling of Rs 10,000
        # would refuse them with "for families earning Rs 10,000 a year or
        # more" — the reverse of the reason they were refused. So the correction
        # takes the sentence with it.
        description = str(r.get("description") or "").strip()
        if corrected:
            description = _rule_sentence(name, op, value)
            issues.append(
                f"rule {name}: its refusal sentence was written for the {proposed} "
                "the model proposed, so it said the opposite of the rule and was "
                "replaced — check the wording")
        elif len(description) < 10 or len(description) > 240:
            description = _rule_sentence(name, op, value)
            issues.append(
                f"rule {name}: the refusal sentence was generated from the rule, "
                "not read from the notice — check the wording")

        out.append({
            "field": name,
            "op": op,
            "value": value,
            # Every rule proposed here is a hard condition: these are the
            # statutory ones — a certificate percentage, an income ceiling —
            # and a soft version of "you need a 40% certificate" means nothing.
            "hard": True,
            "description": description,
        })

    # Two rules on one field would be contradictory more often than not, and the
    # matcher applies both. Keep the first and say so.
    seen: set[str] = set()
    deduped = []
    for r in out:
        if r["field"] in seen:
            issues.append(f"rule {r['field']}: a second rule on the same field was dropped")
            continue
        seen.add(r["field"])
        deduped.append(r)
    return deduped


# How much of a page is worth sending. A stripped scholarship notice is a few
# thousand characters; this is the ceiling for the outlier that inlines its
# whole site in one document, and it is cut at the end rather than the start
# because the eligibility table and the deadline are near the top and the
# footer navigation is not.
MAX_PAGE_CHARS = 120_000

# Below this much actual text, a fetch has not read the page.
#
# The number is not arbitrary and the case is not rare. A single-page app serves
# a shell — a <title>, a <noscript>, and the comments the build tool left — and
# an HTTP GET against it returns a couple of hundred characters that look like a
# successful fetch and contain no scheme. www.sbiashascholarship.co.in is one:
# 351 characters, of which the only real content is the title.
#
# Handing that to the model is worse than not fetching at all. It would answer
# from a title and its own memory, confidently, in the requested shape — and the
# operator would get a draft that looks read rather than an obvious failure. So
# a thin page is treated as no page, and the model browses it instead.
#
# 800 non-whitespace characters: a stripped notice runs to thousands, and the
# thinnest real one is still an order of magnitude above a JavaScript shell.
MIN_PAGE_CHARS = 800


def too_thin(text: str | None) -> bool:
    """Whether a fetched page has enough on it to be worth reading. See above."""
    if not text:
        return True
    return len("".join(text.split())) < MIN_PAGE_CHARS


def extract(url: str, allowed_tags: list[str], api_key: str | None = None,
            page_text: str | None = None) -> Proposal:
    """Read one page into a proposed curated listing.

    `allowed_tags` comes from the backend, which reads it out of the listing_tag
    table. The vocabulary is closed and it is not an enum, so this service has
    no way to know it — and a tag invented here is refused by the API with a
    field map the operator never sees.

    `page_text` is the page, already fetched. It is the difference between a
    read that takes a few seconds and one that takes most of a minute, and the
    reason is that url_context is not a faster way of doing the same thing: a
    call carrying it is the model deciding to browse, issuing a fetch, waiting
    on a government web server, and only then beginning to read. Handed the
    text, none of that is in the call.

    The browse path stays for the pages our own fetch cannot have — a 403 to a
    non-browser user agent, a notice rendered by JavaScript. It is slow and it
    is better than nothing, which is the right order of preference.
    """
    tags = ", ".join(allowed_tags) if allowed_tags else "(none available)"
    # Each field with what a value on it means, rather than with its operator.
    # Naming "LTE" here only invited the model to send an operator back, and the
    # one it sent back was sometimes reversed; _clean_rules fills the comparison
    # in from PROPOSABLE_RULES either way, so the prompt asks for the half the
    # model is actually good at.
    rule_fields = "\n".join(
        f"    {name:<22}{RULE_MEANING[name]}" for name in PROPOSABLE_RULES)

    prompt = f"""Read this scholarship page and describe it: {url}

Today is {date.today().isoformat()}.

It has already been judged a real education scholarship open to students with
disabilities in India. Do not re-judge it — describe it.

Use these exact values where a field is constrained:

  sponsor_type      one of: {', '.join(sorted(SPONSOR_TYPES))}
  award_basis       one of: {', '.join(sorted(AWARD_BASES))}
  tags              from: {tags}
  disability types  from: {', '.join(DISABILITY_TYPES)}
  course levels     from: {', '.join(COURSE_LEVELS)}
  state codes       two-letter codes, e.g. MH, TN, UP, DL

Eligibility rules are what the matcher evaluates, so state the conditions the
notice actually gives, and only those. The available fields, and what a value on
each one means:

{rule_fields}

Give each rule a `field`, a `value` and a `description` — and no comparison. The
comparison is fixed by the field: a disability percentage is always a minimum, an
income figure is always a maximum, and a list is always the set that is accepted.

Do not invent a rule the notice does not state. In particular, do not add a 40%
disability rule unless the notice says so — a scheme silently given the usual
threshold quietly excludes everyone between its real threshold and that one.
Leave the list empty if the notice states no conditions.

For each rule also write `description`: one sentence, 10-240 characters,
addressed to a student who fails it, in the notice's own terms.

Dates are read off the page, never remembered:

  - opens_at and closes_at are YYYY-MM-DD.
  - An Indian notice writes dates day first. 05/06/2026 is 5 June 2026.
  - A day and month with no year mean the next such date that has not passed.
  - If the page states no date, use null. Null is the safe answer: a listing
    with no closing date is shown as open. A year carried over from an earlier
    cycle is not — it removes the scheme from the site altogether.

Answer as JSON:
{{"title": "official name",
  "sponsor_name": "the organisation that runs and funds it",
  "sponsor_type": "GOVERNMENT",
  "summary": "who it is for and what they get, plain words, 20-500 chars",
  "description": "a longer overview",
  "external_url": "the page a student actually applies on",
  "academic_year": "2026-27 or empty",
  "award_basis": "MERIT",
  "award_amount": 50000,
  "opens_at": "YYYY-MM-DD or null",
  "closes_at": "YYYY-MM-DD or null",
  "benefit_summary": "one line, e.g. Full tuition + Rs 3,000 a month",
  "benefit_description": "instalments, what is covered, what is not",
  "eligibility_summary": "the conditions restated for a person to read",
  "application_process": "the steps in order",
  "documents_required": ["Disability certificate", "Income certificate"],
  "important_notes": "quotas, caveats",
  "contact_email": "", "contact_phone": "",
  "tags": ["Engineering"],
  "rules": [{{"field": "disability_percent", "value": 40,
              "description": "This scheme needs a certified disability of 40% or more."}},
            {{"field": "annual_family_income", "value": 250000,
              "description": "This scheme is for families earning \u20b92.5 lakh a year or less."}}]}}

external_url must be where the student applies — the scheme's own site or the
official portal (NSP, AICTE), not an aggregator such as buddy4study.com or
myscheme.gov.in. If the page offers no application link, use {url}.

Convert lakh to digits: 1 lakh = 100000. Rupee figures as plain integers."""

    if page_text:
        # The page instead of an instruction to go and get it. Fenced and
        # labelled so a notice containing the word "JSON", or its own set of
        # instructions, reads as the thing being described rather than as part
        # of the description.
        prompt = (
            prompt.replace(f"Read this scholarship page and describe it: {url}",
                           f"Read the scholarship page below and describe it.\n"
                           f"It was fetched from: {url}")
            + "\n\n----- the page -----\n"
            + page_text[:MAX_PAGE_CHARS]
            + "\n----- end of page -----"
        )

    data = _as_json(_ask(prompt, None if page_text else "url_context", api_key))
    if not isinstance(data, dict):
        raise ModelError("extract did not return an object")

    issues: list[str] = []

    def text(key: str, limit: int) -> str:
        return str(data.get(key) or "").strip()[:limit]

    p = Proposal(
        title=text("title", 200),
        sponsor_name=text("sponsor_name", 200),
        summary=text("summary", 500),
        description=text("description", 20000),
        external_url=text("external_url", 2000) or url,
        academic_year=text("academic_year", 9),
        benefit_summary=text("benefit_summary", 300),
        benefit_description=text("benefit_description", 20000),
        eligibility_summary=text("eligibility_summary", 20000),
        application_process=text("application_process", 20000),
        important_notes=text("important_notes", 20000),
        contact_email=text("contact_email", 200),
        contact_phone=text("contact_phone", 20),
    )

    sponsor_type = text("sponsor_type", 40).upper()
    if sponsor_type in SPONSOR_TYPES:
        p.sponsor_type = sponsor_type
    else:
        # Refused rather than defaulted to GOVERNMENT: the sponsor type is a
        # facet students filter on, and a wrong one is worse than a blank the
        # operator has to fill.
        if sponsor_type:
            issues.append(f"sponsor_type {sponsor_type!r} is not an org_type")

    basis = text("award_basis", 40).upper()
    if basis in AWARD_BASES:
        p.award_basis = basis
    elif basis:
        issues.append(f"award_basis {basis!r} is not one of the five")

    amount = data.get("award_amount")
    if amount is not None:
        try:
            p.award_amount = float(amount)
            if p.award_amount <= 0:
                p.award_amount = None
        except (TypeError, ValueError):
            issues.append(f"award_amount {amount!r} is not a number")

    for key in ("opens_at", "closes_at"):
        raw = data.get(key)
        parsed = _parse_date(raw)
        if parsed:
            setattr(p, key, parsed)
        elif raw:
            issues.append(f"{key} {raw!r} could not be read as a date, and was dropped")

    if p.opens_at and p.closes_at and p.closes_at <= p.opens_at:
        # The API refuses this pair outright, so the weaker of the two goes and
        # the operator is told which.
        issues.append(
            f"closes_at {p.closes_at} is not after opens_at {p.opens_at}; both dropped")
        p.opens_at = p.closes_at = None

    issues.extend(_date_issues(p))

    docs = data.get("documents_required")
    if isinstance(docs, list):
        p.documents_required = [str(d).strip()[:120] for d in docs if str(d).strip()][:20]

    allowed = set(allowed_tags or [])
    raw_tags = data.get("tags")
    if isinstance(raw_tags, list):
        for t in raw_tags:
            t = str(t).strip()
            if t in allowed and t not in p.tags:
                p.tags.append(t)
            elif t and t not in allowed:
                issues.append(f"tag {t!r} is not in the directory's vocabulary")
    p.tags = p.tags[:12]

    p.rules = _clean_rules(data.get("rules"), issues)

    # The three the API will refuse outright, said here so the run reports a
    # reason rather than a 422 the operator cannot see.
    if len(p.title) < 4:
        issues.append("title is missing or shorter than 4 characters")
    if len(p.summary) < 20:
        issues.append("summary is shorter than the 20 characters the directory needs")
    if not p.sponsor_name:
        issues.append("sponsor_name is missing")
    if not p.rules:
        issues.append(
            "no eligibility rules — the listing can be saved as a draft but "
            "cannot be published until one is added")

    p.issues = issues
    return p


# --- stage 4: translate ------------------------------------------------------
#
# What the platform can store is narrower than what this produces, and the gap
# is deliberate. `scholarship` has `summary_hi` and `description_hi` and nothing
# else — two columns and one language, chosen when the site was English and
# Hindi. Encoding that limit here would be this service deciding what the
# catalogue can hold, which is the backend's business and not its own. So this
# takes a list of languages; when a third one has somewhere to live, the backend
# passes another code and nothing in this file changes.

# The code a listing is stored under, and the name the model is asked in.
#
# A bare two-letter code in a prompt is not a reliable request: "or" is Odia and
# also an English word, "as" likewise, and a model that misreads one returns
# fluent text in the wrong language. Naming the language is what removes the
# ambiguity — and a code that is not in this table is refused rather than
# guessed at, because nobody in the review queue reads Odia well enough to catch
# a translation that quietly is not one.
LANGUAGE_NAMES: dict[str, str] = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit", "sd": "Sindhi",
    "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}

# Ceilings, when the caller names none. The backend passes the real ones per
# field, because it is the side that knows the column widths.
DEFAULT_LIMIT = 24000
MAX_LANGUAGES = 8


def translate(fields: dict[str, str], languages: list[str],
              limits: dict[str, int] | None = None,
              api_key: str | None = None) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Translate a listing's student-facing prose into each language asked for.

    A call of its own rather than more of the extract prompt, and that is a
    deliberate ~30-60s per drafted page. Extraction is the expensive thing to
    lose: it has read the page, and its reply already carries twenty fields and
    a rule list. Four languages of prose on top of that makes the reply that
    fails to parse both bigger and likelier, and a failure there costs the
    scholarship, not the translation. Separated, the worst case is an English
    listing — which is what every discovered listing is today.

    Returns the translations and the issues, rather than raising on a partial
    answer. One language coming back unusable should not cost the others, and
    none of this is worth failing a draft over.
    """
    limits = limits or {}
    issues: list[str] = []

    wanted: list[str] = []
    for code in languages:
        code = str(code).strip().lower()
        if code in ("", "en"):
            # English is the original, not a translation of it.
            continue
        if code not in LANGUAGE_NAMES:
            issues.append(f"language {code!r} is not one this service will translate into")
            continue
        if code not in wanted:
            wanted.append(code)

    text = {k: v.strip() for k, v in fields.items() if isinstance(v, str) and v.strip()}
    if not wanted or not text:
        return {}, issues
    if len(wanted) > MAX_LANGUAGES:
        issues.append(f"only the first {MAX_LANGUAGES} languages were translated")
        wanted = wanted[:MAX_LANGUAGES]

    named = ", ".join(f"{LANGUAGE_NAMES[c]} (as \"{c}\")" for c in wanted)
    prompt = f"""Translate the fields below into: {named}

This is a scholarship listing shown to a student with a disability in India who
is deciding whether they can apply. Translate it — do not summarise it, do not
expand it, and do not add anything the English does not say.

Keep unchanged, in every language:
  - rupee figures, percentages, dates and any other number
  - URLs, email addresses and phone numbers
  - the official name of the scheme and of the organisation that runs it,
    which are proper nouns. Where the name is normally written in the script of
    the target language, use that; otherwise leave it in English rather than
    inventing a translation of a name.

Write plainly, addressed to the student, in the register a government notice
would use if it had been written in that language to begin with — not a
transliteration of English sentence structure.

Do not translate the field names. Answer as JSON, one object per language code:

{{{", ".join(f'"{c}": {{...}}' for c in wanted)}}}

Each object has exactly these keys: {", ".join(text)}

The English:

{json.dumps(text, ensure_ascii=False, indent=2)}"""

    data = _as_json(_ask(prompt, None, api_key))
    if not isinstance(data, dict):
        raise ModelError("translate did not return an object")

    out = _take_translations(data, wanted, text, limits, issues)
    return out, issues


def _take_translations(data: dict, wanted: list[str], text: dict[str, str],
                       limits: dict[str, int], issues: list[str]) -> dict[str, dict[str, str]]:
    """Keep what came back that is usable, and say what was not.

    Separate from translate() so it can be tested without a key. Everything
    here is a judgement about a reply this service cannot otherwise reach, and
    it is the half of the stage that has to be right when the model is having a
    bad day.
    """
    out: dict[str, dict[str, str]] = {}
    for code in wanted:
        block = data.get(code)
        if not isinstance(block, dict):
            issues.append(f"{LANGUAGE_NAMES[code]}: the model returned nothing usable")
            continue

        kept: dict[str, str] = {}
        for name, english in text.items():
            got = block.get(name)
            if not isinstance(got, str) or not got.strip():
                continue
            got = got.strip()
            if got == english:
                # Not dropped: a scheme's official name is often left in English
                # on purpose, and so is a URL. Said, because the other reason
                # for it is a model that could not do the job and echoed the
                # input, and those look identical from here.
                issues.append(
                    f"{LANGUAGE_NAMES[code]}: {name} came back identical to the "
                    "English — check whether it was translated at all")
            limit = limits.get(name, DEFAULT_LIMIT)
            if len(got) > limit:
                got = got[:limit]
                issues.append(
                    f"{LANGUAGE_NAMES[code]}: {name} was longer than the {limit} "
                    "characters the column holds, and was cut")
            kept[name] = got

        missing = [n for n in text if n not in kept]
        if missing:
            issues.append(
                f"{LANGUAGE_NAMES[code]}: no translation came back for "
                + ", ".join(missing))
        if kept:
            out[code] = kept

    return out
