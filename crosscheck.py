"""Read the page a second time, with the patterns, and compare.

Why a second reader
-------------------
extract.py exists because of one sentence in its own docstring: "a wrong income
ceiling silently excludes every student above it, and 'the model said so' is not
something you can show a provider or a student." Adding a model does not answer
that. It makes it worse, because the model's answer arrives with no span
attached and reads exactly as confidently when it is wrong.

So the patterns stay, and they are pointed at the same page. Three numbers
decide who is told they qualify — the certificate percentage, the income
ceiling, the closing date — and for each one this produces:

    agree        both readers found the same value. The evidence span from the
                 pattern is attached, so the operator can see the sentence.
    disagree     both found a value and they differ. Named loudly; this is the
                 case that would otherwise ship silently.
    model only   the patterns found nothing. Common and usually fine — the
                 notice phrased it in prose — but unverified, and said so.
    pattern only the model missed something stated in the stereotyped form the
                 patterns are built for. Worth a look.

Nothing here overrides the model. It annotates, because which reader is right
is a judgement about the page, and that is the operator's. The one value handed
back rather than only described is the closing date, because the caller has a
decision to make about it that it cannot make without the value — see
`Check.closes_at`, and `settle_closing_date` in service.py for what is done with
it and why the deadline in particular earns the exception.

The fetch is a second network trip for a page the model already read through
url_context. That is the cost of the check, and it is one HTTP GET against a
notice — cheap next to the two model calls either side of it.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402

UA = "IndicAI-scholarship-discovery/1.0 (+https://scholarship.indic-ai.org)"

# Enough of a notice to hold its eligibility table. A few pages are megabytes of
# inline script; reading all of it to regex 2 KB of prose is waste, and the cap
# is what stops one pathological page stalling a run.
MAX_BYTES = 2_000_000


@dataclass
class Check:
    """What the patterns saw, beside what the model said."""

    # field name -> the sentence the pattern read it from
    evidence: dict[str, str] = field(default_factory=dict)
    # Human-readable, appended to the proposal's issues.
    notes: list[str] = field(default_factory=list)
    # False when the page could not be fetched, so "the patterns found nothing"
    # is not mistaken for "the patterns disagreed".
    ran: bool = True

    # What the patterns read for the deadline, where they read one.
    #
    # A reading, not a verdict. This module still only annotates — which reader
    # is right about a page is a judgement and it is the operator's — but the
    # deadline is the one field where the caller has a decision to make that it
    # cannot make without the value, so the value is handed over. service.py
    # decides what to do with it, and says why.
    closes_at: Optional[str] = None
    # True when the notice gave a day and a month and no year, so the year in
    # `closes_at` was worked out rather than read off the page.
    closes_at_inferred: bool = False


def strip_html(html: str) -> str:
    """HTML to readable text.

    Script and style go first — their contents are text to a regex and noise to
    a reader, and a stray "40%" inside a stylesheet is exactly the kind of thing
    that would be read as a disability threshold. (The same function as
    pipeline.py's, which is where it was written.)
    """
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                         ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"),
                         ("&rsquo;", "'"), ("&ndash;", "-")):
        text = text.replace(entity, char)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def fetch_text(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(MAX_BYTES)
        charset = r.headers.get_content_charset() or "utf-8"
    body = raw.decode(charset, errors="replace")
    return strip_html(body) if "<" in body[:2000] else body


def _rule_value(rules: list[dict], field_name: str):
    for r in rules:
        if r.get("field") == field_name:
            return r.get("value")
    return None


def against(url: str, proposal_rules: list[dict], closes_at: str | None,
            today: date | None = None) -> Check:
    """Compare a proposal's decisive numbers with what the patterns read.

    `today` is the reference for a deadline the notice states without a year,
    passed in so one request reads a page against one date.
    """
    today = today or date.today()
    check = Check()

    try:
        text = fetch_text(url)
    except Exception as e:
        # Not an error for the run. The model read the page through its own
        # tool; this reader simply could not, which is worth recording so the
        # absence of notes is not read as agreement.
        check.ran = False
        check.notes.append(f"not independently checked: the page could not be fetched ({e})")
        return check

    reading = extract.read(text, today)

    comparisons = (
        ("certified disability", "disability_percent",
         reading.min_disability_percentage, _rule_value(proposal_rules, "disability_percent"), "%"),
        ("family income ceiling", "annual_family_income",
         reading.max_family_income, _rule_value(proposal_rules, "annual_family_income"), "₹"),
    )

    for label, key, pattern_field, model_value, unit in comparisons:
        pattern_value = pattern_field.value if pattern_field else None

        if pattern_field:
            check.evidence[key] = pattern_field.evidence

        if pattern_value is None and model_value is None:
            continue
        if pattern_value is None:
            check.notes.append(
                f"{label}: {unit}{model_value} came from the model only — the page "
                "does not state it in a form the patterns read, so it is unverified")
            continue
        if model_value is None:
            check.notes.append(
                f"{label}: the page states {unit}{pattern_value} "
                f"(“{pattern_field.evidence}”) and the proposal has no rule for it")
            continue
        if float(pattern_value) != float(model_value):
            check.notes.append(
                f"{label}: the model said {unit}{model_value}, the page reads "
                f"{unit}{pattern_value} — “{pattern_field.evidence}”. "
                "Check before publishing.")

    # The deadline is not a rule, so it is compared separately — and it is the
    # comparison with the sharpest consequence. A wrong closing date does not
    # show a student "closed": the public directory selects `closes_at > now()`
    # (migration 0043), so a stale year takes the scheme off the site, and a
    # date invented in the future keeps a dead scheme on it.
    if reading.closing_date:
        pattern_date = reading.closing_date.value
        check.evidence["closes_at"] = reading.closing_date.evidence
        check.closes_at = pattern_date
        check.closes_at_inferred = reading.closing_date.inferred

        if reading.closing_date.inferred:
            check.notes.append(
                f"closing date: the notice gives a day and a month and no year "
                f"(“{reading.closing_date.evidence}”), read as {pattern_date} — the "
                "next one that has not passed. Check it against the scheme's cycle.")

        if closes_at and closes_at != pattern_date:
            check.notes.append(
                f"closing date: the model said {closes_at}, the page reads "
                f"{pattern_date} — “{reading.closing_date.evidence}”")
        elif not closes_at:
            check.notes.append(
                f"closing date: the page states {pattern_date} "
                f"(“{reading.closing_date.evidence}”) and the proposal has none")
    elif closes_at and closes_at < today.isoformat():
        # Nothing to compare against, and the one date in play has already gone.
        # Worth saying on its own, because this is the model's characteristic
        # failure — the right day and month with a year from whenever it last
        # saw the page — and it is invisible in the panel: the listing simply
        # never appears in the directory.
        check.notes.append(
            f"closing date: the model said {closes_at}, which has already passed, "
            "and the page states no deadline in a form the patterns read. A past "
            "closing date removes the scheme from the directory rather than "
            "showing it as closed — check the year before publishing.")

    # The opening date, where the notice introduces one in its own words. Read
    # for evidence only: it is not compared, because a notice that states an
    # opening date and a proposal that omits one are not in disagreement, and
    # the year is never inferred for it (see extract.find_opening_date).
    if reading.opening_date:
        check.evidence["opens_at"] = reading.opening_date.evidence

    return check
