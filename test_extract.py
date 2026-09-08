"""Tests for the keyless reader.

Run:  python3 llm_agents/test_extract.py

The text below is written the way real notices are written — Indian digit
grouping, "lakh", "orthopaedically handicapped", a deadline three paragraphs
from an issue date. Each case exists because it is a way of getting the answer
quietly wrong.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402

# Fixed, because half of what follows is about a year the notice did not state.
# A reference date taken from the clock would make these tests a different
# question every year and a failing one every November.
TODAY = date(2026, 9, 7)

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")


TOP_CLASS = """
Top Class Education Scheme for Students with Disabilities

The Department of Empowerment of Persons with Disabilities invites applications
for the academic year 2026-27. The scheme is open to students with 40% or more
disability holding a valid UDID card, who are pursuing graduation or
post-graduation in a notified institution.

Total family income from all sources shall not exceed Rs. 8,00,000 per annum.

Tuition fee up to Rs. 2 lakh per year is reimbursed, along with a monthly
maintenance allowance. Issued on 12 March 2026.

Last date for submission of applications: 31 October 2026.
"""

PRE_MATRIC = """
Pre-Matric Scholarship for Students with Disabilities

For school students of Class 9 and Class 10 who are blind, have low vision, are
deaf or hard of hearing, or are orthopaedically handicapped. Students with
cerebral palsy and specific learning disabilities are also eligible.

Applicants must have a disability of not less than 40 per cent. The annual
income of parents/guardians should be below ₹2,50,000.

Scholarship of ₹12,000 per annum. Available in Maharashtra and Tamil Nadu.
Deadline: 15/09/2026
"""


def main() -> None:
    print("\nTop Class Education Scheme")
    r = extract.read(TOP_CLASS)
    check("income ceiling is the income, not the award", r.max_family_income.value, 800000)
    check("award is the award, not the income", r.benefit_amount_max.value, 200000)
    check("minimum disability percentage", r.min_disability_percentage.value, 40)
    check("deadline, not the issue date", r.closing_date.value, "2026-10-31")
    check("levels", sorted(r.course_levels), ["POSTGRADUATE", "UNDERGRADUATE"])
    check("all-India: no state named", r.state_codes, [])

    print("\nPre-Matric Scholarship")
    r = extract.read(PRE_MATRIC)
    check("lakh-free income with Indian grouping", r.max_family_income.value, 250000)
    check("'not less than 40 per cent'", r.min_disability_percentage.value, 40)
    check("numeric deadline", r.closing_date.value, "2026-09-15")
    check("states", sorted(r.state_codes), ["MH", "TN"])
    check(
        "disability types, in database spelling",
        sorted(r.disability_types),
        ["BLINDNESS", "CEREBRAL_PALSY", "HEARING_IMPAIRMENT", "LOCOMOTOR_DISABILITY",
         "LOW_VISION", "SPECIFIC_LEARNING_DISABILITY"],
    )
    check("school level", r.course_levels, ["SCHOOL"])

    print("\nAmount parsing")
    check("2.5 lakh", extract.rupees("2.5", "lakh"), 250000)
    check("Indian grouping", extract.rupees("2,50,000", None), 250000)
    check("crore", extract.rupees("1", "crore"), 10000000)

    print("\nDates, which are the field with the sharpest consequence")

    def deadline(text, today=TODAY):
        f = extract.find_closing_date(text, today)
        return f.value if f else None

    # The four forms a notice writes a dated deadline in.
    check("ISO, as a portal's own markup gives it",
          deadline("Last date: 2026-10-31 for all applicants."), "2026-10-31")
    check("dotted, which the old numeric pattern could not read",
          deadline("Last date: 31.10.2026"), "2026-10-31")
    check("month first", deadline("Deadline: October 31, 2026"), "2026-10-31")
    check("two-digit year", deadline("Apply before 31/10/26"), "2026-10-31")

    # No year at all, which is common and was previously not read.
    check("no year: the next one that has not passed",
          deadline("Last date for submission: 31st October"), "2026-10-31")
    check("no year, and this year's has gone",
          deadline("Last date for submission: 31st October", date(2026, 11, 30)),
          "2027-10-31")
    check("no year, month first",
          deadline("Applications close October 31"), "2026-10-31")
    check("a year the notice does state is never inferred over",
          deadline("Last date: 31 October 2027"), "2027-10-31")

    # Refusing to answer, which is the right answer more often than it looks.
    check("a date that does not exist is not rolled forward",
          deadline("Last date: 31 February 2026"), None)
    check("an unanchored date is not a deadline",
          deadline("Issued on 12 March 2026 by the Department."), None)
    check("29 February in a common year is refused, not moved",
          deadline("Last date: 29/02/2026"), None)

    # Written the American way. Only catchable when the day is above twelve —
    # a bare 05/06/2026 is genuinely ambiguous and is read day-first.
    check("MM/DD/YYYY is caught when it can be",
          deadline("Apply by 10/31/2026"), "2026-10-31")
    check("ambiguous is read day first, as an Indian notice writes it",
          deadline("Apply by 05/06/2026"), "2026-06-05")

    check("the year, when inferred, is flagged as inferred",
          extract.find_closing_date("Last date: 31st October", TODAY).inferred, True)
    check("a year read off the page is not flagged",
          extract.find_closing_date("Last date: 31 October 2026", TODAY).inferred, False)

    print("\nThe opening date, which is not the deadline reversed")
    opening = extract.find_opening_date("Applications open on 01/08/2026.", TODAY)
    check("an opening date with a year is read", opening.value, "2026-08-01")
    # Never inferred forward: _next_such_day answers "the next one coming",
    # which would hide the listing until then.
    check("an opening date with no year is left alone",
          extract.find_opening_date("Applications open on 1st August", TODAY), None)
    check("the deadline's words do not read as the opening date",
          extract.find_opening_date("Last date: 31 October 2026", TODAY), None)

    print("\nEvidence is carried")
    r = extract.read(TOP_CLASS)
    check("income cites its sentence", "8,00,000" in r.max_family_income.evidence, True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
