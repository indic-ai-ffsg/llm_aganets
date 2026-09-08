"""Generate the pipeline's enums from the database's own enum types.

Run:  python3 llm_agents_call/generate_enums.py

Why this exists
---------------
The discovery pipeline used to declare its own vocabulary — "Deaf and Hearing
Impaired", "Class 9", "PhD / Doctorate" — beside a database that has
HEARING_IMPAIRMENT and four course levels. Nothing extracted could be imported:
every value was either a different spelling of a real one or, for
"Neurodiverse Conditions" and "Wilson's Disease", not a member of the enum at
all, so the INSERT would fail. MULTIPLE_DISABILITIES, which the RPWD Act does
list, was missing entirely.

lib/fields.ts already carries the warning that the wizard and the public
eligibility check have to offer identical values "or the answers would arrive
and be dropped". This is that same failure one layer further out, and the fix is
the same one: stop typing the list twice.

The migration series is the source of truth — 0001's CREATE TYPE plus every
later ALTER TYPE that extends or renames a value. The models are told to answer
in these exact strings, and anything they invent instead is caught by the
validation gate rather than fuzzy-matched into something plausible.
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "backend/internal/platform/postgres/sql"
OUT = Path(__file__).resolve().parent / "enums.py"

# The enum types worth extracting against, and the Python class each becomes.
WANTED = {
    "disability_type": "DisabilityType",
    "course_level": "CourseLevel",
    "social_category": "SocialCategory",
    "org_type": "SponsorType",
}


def read_enums(sql_dir: Path) -> dict[str, list[str]]:
    """Every wanted enum as the migrations leave it, not as 0001 declared it.

    This read the CREATE TYPE in 0001 and stopped, which is wrong for any enum
    a later migration extends — and one is. 0024 adds PRIVATE to org_type with
    ALTER TYPE ... ADD VALUE, so the generated SponsorType held three values
    against a database holding four, and a proposal naming the fourth was
    refused by this pipeline as "not a member" while the API would have taken
    it. The file promises that everything in it is importable by construction;
    reading only the first migration cannot keep that promise.

    Applied in filename order, which is migration order — a RENAME after an ADD
    has to land second or the renamed value comes back.
    """
    found: dict[str, list[str]] = {}

    for path in sorted(sql_dir.glob("*.sql")):
        sql = path.read_text()

        for name, body in re.findall(
            r"CREATE TYPE\s+(\w+)\s+AS ENUM\s*\((.*?)\);", sql, re.S
        ):
            if name in WANTED:
                found[name] = re.findall(r"'([^']+)'", body)

        # `IF NOT EXISTS` is optional in the syntax and present in every one of
        # ours; the value is appended, because that is where Postgres puts it
        # without a BEFORE/AFTER clause.
        for name, value in re.findall(
            r"ALTER TYPE\s+(\w+)\s+ADD VALUE\s+(?:IF NOT EXISTS\s+)?'([^']+)'",
            sql, re.I,
        ):
            if name in WANTED and value not in found.get(name, []):
                found.setdefault(name, []).append(value)

        # A rename keeps the value's position, so this edits in place rather
        # than removing and appending. 0022 and 0025 both rename app_role,
        # which is not wanted here — but the next one might be.
        for name, old, new in re.findall(
            r"ALTER TYPE\s+(\w+)\s+RENAME VALUE\s+'([^']+)'\s+TO\s+'([^']+)'",
            sql, re.I,
        ):
            if name in WANTED and old in found.get(name, []):
                found[name][found[name].index(old)] = new

    return found


def member(value: str) -> str:
    """PARKINSONS_DISEASE -> PARKINSONS_DISEASE; already a valid identifier."""
    return value


def render(found: dict[str, list[str]]) -> str:
    lines = [
        '"""Generated from the migrations in postgres/sql/. Do not edit by hand.',
        "",
        "Regenerate with:  python3 llm_agents_call/generate_enums.py",
        "",
        "Every value here is a member of a Postgres enum, as the whole migration",
        "series leaves it and not as 0001 declared it — org_type gains PRIVATE in",
        "0024. A scholarship extracted with a value that is not in these lists",
        "cannot be imported, so the pipeline rejects it rather than guessing what",
        "was meant.",
        '"""',
        "",
        "from enum import Enum",
        "",
    ]

    for sql_name, class_name in WANTED.items():
        values = found.get(sql_name)
        if not values:
            raise SystemExit(f"enum {sql_name} not found in {SQL_DIR.name}/")
        lines += [f"", f"class {class_name}(str, Enum):"]
        lines += [f'    {member(v)} = "{v}"' for v in values]
        lines.append("")

    lines += [
        "",
        "def values(enum_cls) -> list[str]:",
        "    return [e.value for e in enum_cls]",
        "",
        "",
        "DISABILITY_TYPES = values(DisabilityType)",
        "COURSE_LEVELS = values(CourseLevel)",
        "SOCIAL_CATEGORIES = values(SocialCategory)",
        "SPONSOR_TYPES = values(SponsorType)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write enums.py, or with --check only report whether it is current.

    --check exists because the failure it catches has now happened twice, and
    the second time took the whole service down rather than narrowing it.

    The first was silent: org_type gained PRIVATE in 0024, the generator read
    only 0001, and this pipeline refused a value the API would have accepted.
    The second was not silent at all — social_category was added to WANTED and
    the file was never regenerated, so vocab.py asked enums.py for a name it did
    not have and every module here, the FastAPI app included, failed to import.

    Both are the same fault: a generated file is committed beside the thing it
    is generated from, and nothing compares them. Writing the file is a decision
    somebody has to make; asking whether it is current is not, so it is separated
    out here and can be run anywhere the migrations are on disk — which is the
    monorepo tree, and is where regeneration happens anyway.
    """
    check_only = "--check" in sys.argv[1:]

    if not SQL_DIR.is_dir():
        raise SystemExit(
            f"cannot find {SQL_DIR}.\n"
            "The migrations are in the backend repository; this needs a tree "
            "where both are checked out beside each other.")

    found = read_enums(SQL_DIR)
    fresh = render(found)

    if check_only:
        current = OUT.read_text() if OUT.exists() else ""
        if current == fresh:
            print(f"{OUT.name} is current — {len(found)} enums, "
                  f"{sum(len(v) for v in found.values())} values")
            return
        # A diff rather than "they differ": the answer to "what changed" is the
        # only thing that says whether this is a value the database gained or a
        # hand edit somebody made to a generated file.
        print(f"{OUT.name} is NOT current. What the migrations say it should be:\n")
        sys.stdout.writelines(difflib.unified_diff(
            current.splitlines(keepends=True), fresh.splitlines(keepends=True),
            fromfile=f"{OUT.name} (committed)", tofile=f"{OUT.name} (generated)"))
        raise SystemExit(
            f"\nRun `python3 {Path(__file__).name}` to write it.")

    OUT.write_text(fresh)
    total = sum(len(v) for v in found.values())
    print(f"wrote {OUT.relative_to(ROOT)} — {len(found)} enums, {total} values")
    for name, vals in found.items():
        print(f"  {name}: {len(vals)}")


if __name__ == "__main__":
    main()
