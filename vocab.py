"""The vocabularies a proposal has to be written in, beyond the Postgres enums.

`enums.py` is generated from 0001_extensions_and_enums.sql and covers the four
real enum types. Two more vocabularies decide whether a proposal is importable,
and neither is an enum, which is why they are here and hand-kept:

  state_code    a free `text` column on the profile. A rule written against
                "Delhi" rather than "DL" is accepted by the API, stored, and
                matches nobody — no error, no match. That is the worst failure
                mode this system has, because it looks like a scheme nobody
                qualifies for rather than like a bug.

  rule fields   the left-hand side of an eligibility rule, and the operators
                the API's `oneof` will accept on it.

Both mirror admin/src/pages/scholarship-vocabulary.ts, which carries the same
warning for the same reason. Where this file and that one disagree, that one
wins: it is the editor an operator actually types into.
"""

from __future__ import annotations


def _generated(name: str) -> list[str]:
    """A list out of enums.py, with a failure that names the fix.

    enums.py is generated from the backend's migrations, committed, and read at
    import by every module here. When it falls behind, the symbol is simply not
    there and what anybody sees is

        ImportError: cannot import name 'SOCIAL_CATEGORIES' from 'enums'

    raised three modules deep, from a service that then cannot start at all —
    no route answers, /healthz included, so the failure looks like a dead
    container rather than a stale file.

    That is not hypothetical. social_category was added to generate_enums.py's
    WANTED table and enums.py was never regenerated, so vocab, extract,
    crosscheck, llm, pipeline and the FastAPI app all failed to import, and
    every test in both suites failed to collect.

    Nothing here can repair it — this file has no idea what the missing values
    are, and guessing them is exactly what generating the file exists to
    prevent. What it can do is stop the message being a puzzle.
    """
    import enums

    try:
        return getattr(enums, name)
    except AttributeError:
        raise ImportError(
            f"enums.py does not define {name}. It is generated from the "
            "backend's migrations and has fallen behind them.\n\n"
            "From the monorepo tree, run:\n"
            "    python3 llm_agents_call/generate_enums.py\n\n"
            "`generate_enums.py --check` answers whether it is current without "
            "writing anything."
        ) from None


# The states and union territories by the code stored on a profile.
#
# DH, not DN, for Dadra and Nagar Haveli: the two union territories merged in
# 2020 and the panel's list uses the merged code. extract.py mapped "dadra" to
# DN, which no profile holds, so every Dadra scheme proposed a rule that could
# not match — the failure this file exists to prevent, present in the code that
# warns about it.
STATE_CODES: dict[str, str] = {
    "andaman": "AN", "nicobar": "AN", "andhra": "AP", "arunachal": "AR",
    "assam": "AS", "bihar": "BR", "chandigarh": "CH", "chhattisgarh": "CT",
    "dadra": "DH", "nagar haveli": "DH", "daman": "DH", "diu": "DH",
    "delhi": "DL", "goa": "GA", "gujarat": "GJ", "haryana": "HR",
    "himachal": "HP", "jammu": "JK", "kashmir": "JK", "jharkhand": "JH",
    "karnataka": "KA", "kerala": "KL", "ladakh": "LA", "lakshadweep": "LD",
    "madhya pradesh": "MP", "maharashtra": "MH", "manipur": "MN",
    "meghalaya": "ML", "mizoram": "MZ", "nagaland": "NL", "odisha": "OR",
    "orissa": "OR", "puducherry": "PY", "pondicherry": "PY", "punjab": "PB",
    "rajasthan": "RJ", "sikkim": "SK", "tamil nadu": "TN", "telangana": "TG",
    "tripura": "TR", "uttar pradesh": "UP", "uttarakhand": "UT",
    "west bengal": "WB",
}

VALID_STATE_CODES: frozenset[str] = frozenset(STATE_CODES.values())

# The operators registry.RuleInput's `oneof` accepts. A proposal carrying
# anything else is refused by the API with a field map, so it is refused here
# first, where the reason can name the vocabulary.
RULE_OPS: frozenset[str] = frozenset({
    "EQ", "NEQ", "GT", "GTE", "LT", "LTE", "IN", "NOT_IN", "BETWEEN", "EXISTS",
})

# The rule fields worth proposing from a notice, and the operator each takes.
#
# Deliberately a subset of RULE_FIELDS. academic_percentage, current_year, age
# and institution_type are all real fields and all things a notice states, but
# they are stated loosely ("good academic record", "students in the final
# year") and a rule guessed from that wording blocks students the scheme would
# have taken. Those belong in eligibility_summary for a person to read, not in
# the engine — so the model is not offered them.
PROPOSABLE_RULES: dict[str, str] = {
    "disability_percent": "GTE",
    "disability_type": "IN",
    "annual_family_income": "LTE",
    "course_level": "IN",
    "state_code": "IN",
    "gender": "IN",
    "social_category": "IN",
}

# What a value on each of those fields means, in the terms a notice uses.
#
# This is the prompt's half of the table above: that one fixes the comparison,
# and this says what the comparison is for. They are separate because they are
# read by different things — the matcher needs the operator, the model needs the
# meaning — and identical in their keys, which is asserted below.
#
# The model is given this and not the operator. Handed "annual_family_income
# (LTE)" it would sometimes answer GTE, and a reversed income comparison is the
# worst failure this pipeline has: a scheme for families under ₹2.5 lakh becomes
# a scheme for families over it. Nothing downstream reads as wrong — the rule is
# well-formed, the API takes it, the matcher evaluates it — and the students it
# was written for are the ones it turns away.
RULE_MEANING: dict[str, str] = {
    "disability_percent":
        "the minimum certified percentage the notice requires — a floor",
    "disability_type":
        "the conditions accepted; leave it out if the notice accepts any",
    "annual_family_income":
        "the highest annual family income allowed — a ceiling, in whole rupees",
    "course_level":
        "the levels of study accepted",
    "state_code":
        "the states of domicile accepted, as two-letter codes",
    "gender":
        "the genders accepted; leave it out for a scheme open to everyone",
    "social_category":
        "the categories accepted; leave it out if the notice accepts any",
}

# What the engine reads each choice field against, so a proposal can be checked
# before it is ever sent. Filled from enums.py at import rather than retyped —
# the point of generating that file is that this list cannot drift from the
# database.
def _choice_domains() -> dict[str, frozenset[str]]:
    return {
        "disability_type": frozenset(_generated("DISABILITY_TYPES")),
        "course_level": frozenset(_generated("COURSE_LEVELS")),
        "social_category": frozenset(_generated("SOCIAL_CATEGORIES")),
        "state_code": VALID_STATE_CODES,
        # Not an enum in enums.py because no generated type covers it: the
        # profile column is `gender` and these are the four stored values.
        "gender": frozenset({"MALE", "FEMALE", "TRANSGENDER", "UNDISCLOSED"}),
    }


CHOICE_DOMAINS: dict[str, frozenset[str]] = _choice_domains()

# MERIT, NEED, MERIT_CUM_MEANS, CATEGORY, OTHER — scholarship.award_basis.
# Not generated: award_basis is a CHECK constraint rather than an enum type, so
# generate_enums.py has nothing to read it from.
AWARD_BASES: frozenset[str] = frozenset({
    "MERIT", "NEED", "MERIT_CUM_MEANS", "CATEGORY", "OTHER",
})


def _sponsor_types() -> frozenset[str]:
    """scholarship.sponsor_type is the org_type enum, so it is generated.

    Written this way rather than as a literal because org_type has already
    grown once — 0024 added PRIVATE — and a literal here would be the third
    place that value has to be remembered.
    """
    return frozenset(_generated("SPONSOR_TYPES"))


SPONSOR_TYPES: frozenset[str] = _sponsor_types()


# --- what has to hold between the tables above ---------------------------------
#
# Checked at import, like extract.py's own vocabulary assertions and for the
# same reason: these tables are edited by hand, by different hands, and every
# way they can disagree produces a proposal that is refused or, worse, quietly
# wrong. Failing here costs a container start; failing later costs a listing.

_missing_meaning = set(PROPOSABLE_RULES) ^ set(RULE_MEANING)
if _missing_meaning:
    raise ImportError(
        "PROPOSABLE_RULES and RULE_MEANING must describe the same fields; these "
        f"are in one and not the other: {', '.join(sorted(_missing_meaning))}")

_bad_ops = {f: op for f, op in PROPOSABLE_RULES.items() if op not in RULE_OPS}
if _bad_ops:
    raise ImportError(
        "PROPOSABLE_RULES declares operators the API will not accept: "
        + ", ".join(f"{f}={op}" for f, op in sorted(_bad_ops.items())))

# A choice field with no domain cannot be validated, so its values would reach
# the API unchecked — which is the one thing this file exists to prevent.
_undomained = {
    f for f, op in PROPOSABLE_RULES.items()
    if op in ("IN", "NOT_IN") and f not in CHOICE_DOMAINS
}
if _undomained:
    raise ImportError(
        "these fields take a set of values and have no domain to check them "
        f"against: {', '.join(sorted(_undomained))}")
