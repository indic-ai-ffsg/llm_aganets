"""Read a scholarship notice without a model, without a key, without a bill.

The case for doing it this way
------------------------------
The fields that decide whether a disabled student is told they qualify for
money are a small, closed set: a disability percentage, an income ceiling, a
deadline, a list of RPWD categories, a state, a level of study. In Indian
scholarship notices these are not free prose — they are stereotyped:

    "students with 40% or more disability"
    "family income not exceeding Rs. 2.5 lakh per annum"
    "Last date for submission: 31 October 2026"

A pattern reads those exactly, the same way every time, and can say which
sentence it read them from. That last part is why this is not a downgrade from
a model for this particular job: a wrong income ceiling silently excludes every
student above it, and "the model said so" is not something you can show a
provider or a student. Every field here comes back with the span it came from.

What it cannot do is judge. Whether a page is a scholarship at all, or is still
open, is left to the operator reviewing the proposal — see pipeline.py. This
module answers "what does this text say", never "is this true".

The vocabularies are generated from the database enums (generate_enums.py), so
an extracted value is importable by construction.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enums import DisabilityType, CourseLevel  # noqa: E402
# The codes live in vocab.py, not here.
#
# There were two tables and they had already diverged: this one mapped
# "dadra" to DN, a code no profile holds since the 2020 merger, so a Dadra
# scheme proposed a rule that could never match. One table cannot disagree
# with itself, which is the same argument enums.py makes one layer down.
from vocab import STATE_CODES as STATE_WORDS  # noqa: E402


@dataclass
class Field_:
    """A value and the words it was read from.

    The evidence is the point. A reviewer approving an income ceiling of
    ₹2,50,000 can see the sentence it came from without opening the source.
    """
    value: object
    evidence: str
    # True when part of the value was not on the page. Only dates set it: a
    # notice reading "last date 31st October" with no year has been read
    # correctly and then completed by inference, and those are two different
    # claims to be making to an operator.
    inferred: bool = False


@dataclass
class Reading:
    disability_types: list[str] = field(default_factory=list)
    course_levels: list[str] = field(default_factory=list)
    state_codes: list[str] = field(default_factory=list)
    min_disability_percentage: Optional[Field_] = None
    max_family_income: Optional[Field_] = None
    opening_date: Optional[Field_] = None
    closing_date: Optional[Field_] = None
    benefit_amount_max: Optional[Field_] = None
    evidence: dict[str, str] = field(default_factory=dict)


# --- the vocabulary ----------------------------------------------------------
#
# The words a notice actually uses, mapped to the enum member they mean. This
# table is the domain knowledge, and it is the part worth owning: "orthopaedically
# handicapped" is what a 2011 circular says for LOCOMOTOR_DISABILITY, and no
# amount of general language ability substitutes for knowing that.

DISABILITY_WORDS: dict[str, list[str]] = {
    "BLINDNESS": ["blind", "blindness", "visually handicapped", "totally blind"],
    "LOW_VISION": ["low vision", "partially sighted", "visually impaired"],
    "HEARING_IMPAIRMENT": ["re:hearing impair\\w*", "hard of hearing", "deaf", "hearing handicapped"],
    "SPEECH_AND_LANGUAGE_DISABILITY": ["speech and language", "re:speech impair\\w*", "mute"],
    "LOCOMOTOR_DISABILITY": [
        "locomotor", "orthopaedically handicapped", "orthopedically handicapped",
        "physically handicapped", "physical disability",
    ],
    "LEPROSY_CURED": ["leprosy cured", "leprosy-cured", "cured leprosy"],
    "DWARFISM": ["dwarfism", "dwarf"],
    "INTELLECTUAL_DISABILITY": ["intellectual disability", "mental retardation"],
    "MENTAL_ILLNESS": ["mental illness", "mentally ill"],
    "AUTISM_SPECTRUM_DISORDER": ["autism", "autistic", "asd"],
    "CEREBRAL_PALSY": ["cerebral palsy"],
    "MUSCULAR_DYSTROPHY": ["muscular dystrophy"],
    "CHRONIC_NEUROLOGICAL_CONDITION": ["chronic neurological"],
    "SPECIFIC_LEARNING_DISABILITY": ["re:learning disabilit\\w*", "dyslexia", "dyscalculia"],
    "MULTIPLE_SCLEROSIS": ["multiple sclerosis"],
    "THALASSEMIA": ["thalassemia", "thalassaemia"],
    "HAEMOPHILIA": ["haemophilia", "hemophilia"],
    "SICKLE_CELL_DISEASE": ["sickle cell"],
    "MULTIPLE_DISABILITIES": ["re:multiple disabilit\\w*", "deafblind", "deaf-blind"],
    "ACID_ATTACK_VICTIM": ["acid attack"],
    "PARKINSONS_DISEASE": ["parkinson"],
}

COURSE_WORDS: dict[str, list[str]] = {
    "SCHOOL": ["class 9", "class 10", "class 11", "class 12", "school student",
               "pre-matric", "post-matric", "secondary", "higher secondary"],
    # "graduation" is guarded: "post-graduation" contains it, and matching the
    # inner word would file every PG scheme as UG as well.
    "UNDERGRADUATE": ["undergraduate", "under-graduate",
                      "re:(?<!post)(?<!post-)(?<!post )graduation",
                      "bachelor", "b.tech", "b.a.", "b.sc", "b.com", "diploma", "iti"],
    "POSTGRADUATE": ["postgraduate", "post-graduate", "post graduation", "post-graduation",
                     "master", "m.tech", "m.a.", "m.sc", "mba"],
    "DOCTORAL": ["doctoral", "ph.d", "phd", "doctorate", "research fellow"],
}


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})


def clean(text: str) -> str:
    """Normalise the whitespace and the punctuation a PDF-to-text leaves behind."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"[ \t ]+", " ", text)


def _span(text: str, m: re.Match, pad: int = 60) -> str:
    lo = max(0, m.start() - pad)
    hi = min(len(text), m.end() + pad)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


# --- amounts -----------------------------------------------------------------

def rupees(amount: str, unit: str | None) -> int:
    """A rupee figure as an integer.

    Indian digit grouping (2,50,000) and the lakh/crore multipliers are the two
    things a naive parser gets wrong, and both appear in nearly every notice.
    """
    n = float(amount.replace(",", ""))
    if unit:
        u = unit.lower()
        if u.startswith("lakh") or u.startswith("lac"):
            n *= 100_000
        elif u.startswith("crore"):
            n *= 10_000_000
    return int(round(n))


AMOUNT = r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)?"


def find_income(text: str) -> Optional[Field_]:
    """The family income ceiling.

    Anchored on the word "income" and on the direction: a ceiling is "not
    exceeding" / "up to" / "less than". A notice also states an *award* in
    rupees, and reading that as the income limit would silently exclude every
    applicant — so the anchor is not optional.
    """
    for m in re.finditer(
        r"(?:annual\s+)?(?:family\s+|parental\s+|parents?'?\s+)?income[^.]{0,80}?"
        r"(?:not\s+exceed(?:ing)?|less\s+than|up\s?to|below|maximum\s+of|ceiling\s+of|upto)"
        r"[^.]{0,30}?" + AMOUNT,
        text, re.I,
    ):
        return Field_(rupees(m.group(1), m.group(2)), _span(text, m))
    # The reversed order: "Rs 2.5 lakh per annum family income"
    for m in re.finditer(AMOUNT + r"[^.]{0,40}?(?:per\s+annum\s+)?(?:family\s+)?income", text, re.I):
        return Field_(rupees(m.group(1), m.group(2)), _span(text, m))
    return None


def find_percentage(text: str) -> Optional[Field_]:
    """The minimum disability percentage — 40% in most schemes, 75% in some."""
    for m in re.finditer(
        r"(\d{1,3})\s*(?:%|per\s?cent(?:age)?)[^.]{0,40}?(?:or\s+more\s+)?disab",
        text, re.I,
    ):
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return Field_(pct, _span(text, m))
    for m in re.finditer(
        r"disab[^.]{0,60}?(?:of\s+|not\s+less\s+than\s+|minimum\s+)?(\d{1,3})\s*(?:%|per\s?cent)",
        text, re.I,
    ):
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return Field_(pct, _span(text, m))
    return None


# The words a notice introduces each date with.
#
# Split, because the two dates are not symmetrical and the difference matters
# below: a deadline with no year is the next one coming, and an opening date
# with no year is usually behind us.
CLOSING_WORDS = (r"(?:last\s+date|last\s+day|closing\s+date|deadline|due\s+date"
                 r"|apply\s+(?:on\s+or\s+)?before|apply\s+by"
                 r"|close[sd]?\s+on|applications?\s+close)")

OPENING_WORDS = (r"(?:opening\s+date|start(?:ing)?\s+date|date\s+of\s+commencement"
                 r"|applications?\s+open|opens?\s+on|apply\s+from"
                 r"|registration\s+(?:starts|begins|opens))")


def _calendar(y: int, mon: int, d: int) -> Optional[date]:
    """The date, or None when the notice states one that does not exist.

    "31 February" and "31/09" are typos a scholarship PDF really does contain,
    and a parser that rolls them forward invents a deadline nobody wrote.
    """
    try:
        return date(y, mon, d)
    except ValueError:
        return None


def _next_such_day(mon: int, d: int, today: date) -> Optional[date]:
    """The year a notice left out of a deadline.

    "Last date: 31st October" means the next 31st October there is, because a
    deadline is a thing in the future. Assuming the current year instead would
    turn every notice read in November into one that closed last month — and a
    closes_at in the past does not show a student "closed" on this platform, it
    removes the scheme from the directory (migration 0043).
    """
    for y in (today.year, today.year + 1):
        got = _calendar(y, mon, d)
        if got and got >= today:
            return got
    return None


def _dmy(a: str, b: str, c: str) -> Optional[date]:
    """A d/m/y date, with the Indian order assumed and the American one caught.

    DD/MM/YYYY is what an Indian notice writes, so that is the reading. A number
    above twelve in the month position settles it either way round; a page
    written MM/DD/YYYY is caught only when its day is above twelve, which is the
    best a bare 05/06/2026 allows. The evidence span travels with the value, so
    an operator can see which it was.
    """
    d, mon, y = int(a), int(b), int(c)
    if y < 100:
        y += 2000
    if mon > 12 and d <= 12:
        d, mon = mon, d
    return _calendar(y, mon, d)


def _date_forms() -> list[tuple[str, object, bool]]:
    """The written forms, in the order they have to be tried.

    The ones carrying a year come first, and that ordering is the point:
    "31/10" is a prefix of "31/10/2026" and "31 October" is a prefix of
    "31 October 2026", so reading a short form first would infer a year over one
    the notice actually stated.

    Every reader takes (groups, today) and returns a date or None. None is the
    right answer far more often than a plausible date is — see the note above
    CLOSING_WORDS about what a wrong date costs.
    """
    months = "|".join(sorted(MONTHS, key=len, reverse=True))
    day = r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:of\s+)?"
    month = r"(" + months + r")[a-z]*\.?,?\s*"

    return [
        # 2026-10-31 — what a portal's own markup gives.
        (r"(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)",
         lambda g, _t: _calendar(int(g[0]), int(g[1]), int(g[2])), False),
        # 31st October 2026
        (day + month + r"(\d{4})(?!\d)",
         lambda g, _t: _calendar(int(g[2]), MONTHS[g[1].lower()[:3]], int(g[0])), False),
        # October 31, 2026 — the comma is between the day and the year here,
        # which is the one place `day` does not already allow for it.
        (month + day + r",?\s*(\d{4})(?!\d)",
         lambda g, _t: _calendar(int(g[2]), MONTHS[g[0].lower()[:3]], int(g[1])), False),
        # 31/10/2026, 31-10-2026, 31.10.2026, 31/10/26
        (r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{2,4})(?!\d)",
         lambda g, _t: _dmy(g[0], g[1], g[2]), False),
        # 31st October, with no year at all — common in Indian notices, and the
        # reason these functions take a reference date. The lookahead keeps this
        # off a year the form above would have read.
        (day + r"(" + months + r")[a-z]*\.?(?!\s*,?\s*\d)",
         lambda g, t: _next_such_day(MONTHS[g[1].lower()[:3]], int(g[0]), t), True),
        # October 31, with no year
        (r"(" + months + r")[a-z]*\.?,?\s*(\d{1,2})(?:st|nd|rd|th)?(?!\s*,?\s*\d)",
         lambda g, t: _next_such_day(MONTHS[g[0].lower()[:3]], int(g[1]), t), True),
    ]


def _find_date(text: str, anchor: str, today: date, infer_year: bool) -> Optional[Field_]:
    """A date, anchored on the words a notice uses to introduce it.

    Unanchored, the first date on the page is as likely to be the date the
    circular was issued — and a deadline read from the wrong line closes a
    scheme that is open, or opens one that has closed.

    The gap between the anchor and the date excludes a full stop, so a match
    cannot run out of one sentence and into an unrelated date in the next.
    """
    for pattern, reader, inferred in _date_forms():
        if inferred and not infer_year:
            continue
        for m in re.finditer(anchor + r"[^.\n]{0,60}?" + pattern, text, re.I):
            got = reader(m.groups(), today)
            if got:
                return Field_(got.isoformat(), _span(text, m), inferred)
    return None


def find_closing_date(text: str, today: Optional[date] = None) -> Optional[Field_]:
    """The deadline."""
    return _find_date(text, CLOSING_WORDS, today or date.today(), infer_year=True)


def find_opening_date(text: str, today: Optional[date] = None) -> Optional[Field_]:
    """The date applications open, where the notice introduces one separately.

    Worth reading for the same reason as the deadline and with the sign
    reversed: the directory hides a listing whose opens_at has not arrived, so
    an opening date landed a year late is a scheme nobody can see.

    A year is never inferred for it. `_next_such_day` answers "the next one
    coming", which is right for a deadline and wrong here — an opening date is
    usually behind us, and guessing it forward would hide the listing for a
    year. Without a year on the page, this returns nothing.
    """
    return _find_date(text, OPENING_WORDS, today or date.today(), infer_year=False)


def find_award(text: str) -> Optional[Field_]:
    """The largest rupee figure that is not the income ceiling."""
    income = find_income(text)
    best: Optional[Field_] = None
    for m in re.finditer(AMOUNT, text, re.I):
        span = _span(text, m)
        if re.search(r"income", span, re.I):
            continue
        amount = rupees(m.group(1), m.group(2))
        if income and amount == income.value:
            continue
        if best is None or amount > int(best.value):
            best = Field_(amount, span)
    return best


def _term(word: str) -> str:
    """A vocabulary entry as a regex, bounded so it cannot match inside a word.

    Bare substring search is what made "iti" match "disabilities" and file every
    school scheme as undergraduate too. The boundaries are lookarounds rather
    than \\b because several terms end in a full stop ("b.a."), where \\b sits in
    the wrong place. An entry may also be given as an explicit regex with an
    "re:" prefix, which is how "graduation" excludes "post-graduation".
    """
    if word.startswith("re:"):
        return word[3:]
    return r"(?<!\w)" + re.escape(word) + r"(?!\w)"


def find_words(text: str, table: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    hits, evidence = [], {}
    for value, words in table.items():
        for w in words:
            m = re.search(_term(w), text, re.I)
            if m:
                hits.append(value)
                evidence[value] = _span(text, m, pad=50)
                break
    return hits, evidence


def find_states(text: str) -> tuple[list[str], dict[str, str]]:
    hits, evidence = [], {}
    for name, code in STATE_WORDS.items():
        m = re.search(_term(name), text, re.I)
        if m and code not in hits:
            hits.append(code)
            evidence[code] = _span(text, m, pad=50)
    return hits, evidence


def read(text: str, today: Optional[date] = None) -> Reading:
    """Everything this notice says, with the words it said it in.

    `today` is the reference for a date the notice states without a year. It is
    a parameter rather than a call to date.today() inside the reader so that a
    test is not a different answer in November, and so the service reads a page
    against one date for the whole of a request.
    """
    text = clean(text)
    today = today or date.today()
    r = Reading()

    r.disability_types, ev_d = find_words(text, DISABILITY_WORDS)
    r.course_levels, ev_c = find_words(text, COURSE_WORDS)
    r.state_codes, ev_s = find_states(text)
    r.evidence = {**ev_d, **ev_c, **ev_s}

    r.min_disability_percentage = find_percentage(text)
    r.max_family_income = find_income(text)
    r.opening_date = find_opening_date(text, today)
    r.closing_date = find_closing_date(text, today)
    r.benefit_amount_max = find_award(text)
    return r


# Both vocabularies are keyed by enum member, so a typo here is a startup error
# rather than a value that quietly never matches.
assert set(DISABILITY_WORDS) == {e.value for e in DisabilityType}, "disability table drifted"
assert set(COURSE_WORDS) == {e.value for e in CourseLevel}, "course table drifted"
