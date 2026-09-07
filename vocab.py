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

# What the engine reads each choice field against, so a proposal can be checked
# before it is ever sent. Filled from enums.py at import rather than retyped —
# the point of generating that file is that this list cannot drift from the
# database.
def _choice_domains() -> dict[str, frozenset[str]]:
    from enums import COURSE_LEVELS, DISABILITY_TYPES, SOCIAL_CATEGORIES

    return {
        "disability_type": frozenset(DISABILITY_TYPES),
        "course_level": frozenset(COURSE_LEVELS),
        "social_category": frozenset(SOCIAL_CATEGORIES),
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
    from enums import SPONSOR_TYPES as generated

    return frozenset(generated)


SPONSOR_TYPES: frozenset[str] = _sponsor_types()
