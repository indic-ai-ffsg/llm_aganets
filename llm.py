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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enums import COURSE_LEVELS, DISABILITY_TYPES  # noqa: E402
from vocab import (  # noqa: E402
    AWARD_BASES, CHOICE_DOMAINS, PROPOSABLE_RULES, RULE_OPS, SPONSOR_TYPES,
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


def _ask(prompt: str, tool: str, api_key: str | None) -> str:
    """One turn, with one tool, returning raw text.

    No response_schema: a JSON schema and a search/browse tool cannot both be
    set on the same call, so the shape is asked for in the prompt and parsed
    defensively below. That is the trade the tools force, and it is why
    _as_json exists rather than trusting the transport.
    """
    from google.genai import types

    client = _client(api_key)
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[_tool(tool)],
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


# --- stage 1: search ---------------------------------------------------------

def search(query: str, api_key: str | None = None) -> list[str]:
    """Candidate URLs for one query.

    The prompt's exclusions are the ones that cost review time: an aggregator
    listing twenty schemes produces one draft naming none of them, and a news
    article about a scheme is not the scheme.
    """
    prompt = f"""Search for: {query}

Find URLs of specific scholarship programmes available to students with
disabilities in India.

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


def classify(url: str, api_key: str | None = None) -> Verdict:
    """Whether this page is a live education scholarship for disabled students.

    The judgement the keyless pipeline refused to make, and the reason it
    refused still applies — so the answer is not final. A page that passes here
    becomes a DRAFT for an operator to approve, never a published listing.
    """
    prompt = f"""Analyse this URL: {url}

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
   - A deadline clearly passed with no sign of a new cycle: not open.
   - A recurring annual scholarship with no dates stated: treat as open.
   - Discontinued: not open.

Answer as JSON:
{{"is_scholarship": true, "is_open": true, "reason": "short reason if either is false"}}"""

    data = _as_json(_ask(prompt, "url_context", api_key))
    if not isinstance(data, dict):
        raise ModelError("classify did not return an object")
    return Verdict(
        is_scholarship=bool(data.get("is_scholarship")),
        is_open=bool(data.get("is_open")),
        reason=str(data.get("reason") or "")[:300],
    )


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
                f"₹{int(value):,} or less.")
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

        op = str(r.get("op") or PROPOSABLE_RULES[name]).strip().upper()
        if op not in RULE_OPS:
            issues.append(f"rule dropped: {name} has operator {op!r}")
            continue

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

        description = str(r.get("description") or "").strip()
        if len(description) < 10 or len(description) > 240:
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


def extract(url: str, allowed_tags: list[str], api_key: str | None = None) -> Proposal:
    """Read one page into a proposed curated listing.

    `allowed_tags` comes from the backend, which reads it out of the listing_tag
    table. The vocabulary is closed and it is not an enum, so this service has
    no way to know it — and a tag invented here is refused by the API with a
    field map the operator never sees.
    """
    tags = ", ".join(allowed_tags) if allowed_tags else "(none available)"
    rule_fields = ", ".join(f"{k} ({v})" for k, v in PROPOSABLE_RULES.items())

    prompt = f"""Read this scholarship page and describe it: {url}

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
notice actually gives, and only those. Available fields, with the operator each
takes: {rule_fields}

Do not invent a rule the notice does not state. In particular, do not add a 40%
disability rule unless the notice says so — a scheme silently given the usual
threshold quietly excludes everyone between its real threshold and that one.
Leave the list empty if the notice states no conditions.

For each rule also write `description`: one sentence, 10-240 characters,
addressed to a student who fails it, in the notice's own terms.

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
  "rules": [{{"field": "disability_percent", "op": "GTE", "value": 40,
              "description": "This scheme needs a certified disability of 40% or more."}}]}}

external_url must be where the student applies — the scheme's own site or the
official portal (NSP, AICTE), not an aggregator such as buddy4study.com or
myscheme.gov.in. If the page offers no application link, use {url}.

Convert lakh to digits: 1 lakh = 100000. Rupee figures as plain integers."""

    data = _as_json(_ask(prompt, "url_context", api_key))
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
        if isinstance(raw, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw.strip()):
            setattr(p, key, raw.strip())
        elif raw:
            issues.append(f"{key} {raw!r} is not a YYYY-MM-DD date and was dropped")

    if p.opens_at and p.closes_at and p.closes_at <= p.opens_at:
        # The API refuses this pair outright, so the weaker of the two goes and
        # the operator is told which.
        issues.append(
            f"closes_at {p.closes_at} is not after opens_at {p.opens_at}; both dropped")
        p.opens_at = p.closes_at = None

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
