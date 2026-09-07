"""Tests for the keyless reader.

Run:  python3 llm_agents/test_extract.py

The text below is written the way real notices are written — Indian digit
grouping, "lakh", "orthopaedically handicapped", a deadline three paragraphs
from an issue date. Each case exists because it is a way of getting the answer
quietly wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402

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

    print("\nEvidence is carried")
    r = extract.read(TOP_CLASS)
    check("income cites its sentence", "8,00,000" in r.max_family_income.evidence, True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
