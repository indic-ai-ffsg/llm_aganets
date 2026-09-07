"""Scholarship discovery, with no model and no key.

    python3 llm_agents/pipeline.py --dir notices/     # a folder of saved pages
    python3 llm_agents/pipeline.py --url https://...  # fetch and read one

What this is
------------
A source is fetched, stripped to text, and read by extract.py. What comes out is
a *proposal*: the fields it found, the sentence each one came from, and a list
of everything it could not answer. An operator approves it before any of it
becomes an eligibility_rule.

There is no classifier deciding whether a page is a scholarship, and that is on
purpose. That judgement is the one a person should be making — the fields are
mechanical, the "is this real and is it open" question is not — and it is also
the one an LLM was least able to justify. The proposal carries its evidence, so
approving it takes seconds and the reviewer can see what they are approving.

Nothing here calls out to a paid service. The only network access is fetching
the page the operator pointed it at.

Carried over from the version that did use a model, because they were bugs
regardless of who does the reading:

  - records that fail validation are rejected, not saved with a warning printed
  - the seen-URL set persists between runs instead of re-doing the work
  - failures are collected and reported instead of swallowed by `except: pass`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402
from enums import COURSE_LEVELS, DISABILITY_TYPES  # noqa: E402

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
UA = "IndicAI-scholarship-discovery/1.0 (+https://indic-ai.org)"


# --- getting to text ---------------------------------------------------------

def strip_html(html: str) -> str:
    """HTML to readable text, without a parser dependency.

    Script and style go first — their contents are text to a regex and noise to
    a reader, and a stray "40%" inside a stylesheet is exactly the kind of thing
    that would be read as a disability threshold.
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


def fetch(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    charset = r.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


# --- the proposal ------------------------------------------------------------

@dataclass
class Proposal:
    source: str
    name: str
    disability_types: list[str] = field(default_factory=list)
    course_levels: list[str] = field(default_factory=list)
    state_codes: list[str] = field(default_factory=list)
    min_disability_percentage: int | None = None
    max_family_income: int | None = None
    closing_date: str | None = None
    benefit_amount_max: int | None = None
    # What the reader could not answer, for the operator to fill in.
    unanswered: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    review: str = "PENDING"
    read_at: str = ""


def title_of(text: str, url: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 140 and re.search(r"[a-z]", line):
            return line
    return urlparse(url).path.rsplit("/", 1)[-1] or url


def propose(text: str, source: str) -> Proposal:
    r = extract.read(text)

    def val(f):
        return f.value if f else None

    p = Proposal(
        source=source,
        name=title_of(text, source),
        disability_types=r.disability_types,
        course_levels=r.course_levels,
        state_codes=r.state_codes,
        min_disability_percentage=val(r.min_disability_percentage),
        max_family_income=val(r.max_family_income),
        closing_date=val(r.closing_date),
        benefit_amount_max=val(r.benefit_amount_max),
        read_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    p.evidence = dict(r.evidence)
    for key, f in (("min_disability_percentage", r.min_disability_percentage),
                   ("max_family_income", r.max_family_income),
                   ("closing_date", r.closing_date),
                   ("benefit_amount_max", r.benefit_amount_max)):
        if f:
            p.evidence[key] = f.evidence
        else:
            # Said plainly rather than defaulted. A scheme silently given the
            # usual 40% threshold is a scheme that quietly excludes anyone
            # between its real threshold and that one.
            p.unanswered.append(key)

    if not p.disability_types:
        p.unanswered.append("disability_types (may genuinely be open to all)")
    if not p.course_levels:
        p.unanswered.append("course_levels (may genuinely be open to all)")
    return p


def problems(p: Proposal) -> list[str]:
    """Everything that would stop this being imported. Strict on purpose."""
    out = []
    for v in p.disability_types:
        if v not in DISABILITY_TYPES:
            out.append(f"disability_types: {v!r} is not a disability_type")
    for v in p.course_levels:
        if v not in COURSE_LEVELS:
            out.append(f"course_levels: {v!r} is not a course_level")
    if p.min_disability_percentage is not None and not (0 <= p.min_disability_percentage <= 100):
        out.append(f"min_disability_percentage: {p.min_disability_percentage} outside 0-100")
    if p.max_family_income is not None and p.max_family_income < 0:
        out.append("max_family_income: negative")
    if not p.name:
        out.append("name: empty")
    return out


# --- dedup across runs -------------------------------------------------------

def canonical(url: str) -> str:
    u = urlparse(url.strip().lower())
    return f"{u.scheme}://{u.netloc.replace('www.', '')}{u.path.rstrip('/')}"


class Seen:
    """Sources already read, remembered between runs."""

    def __init__(self, path: Path):
        self.path = path
        self.urls: set[str] = set(json.loads(path.read_text())) if path.exists() else set()

    def add(self, url: str) -> None:
        self.urls.add(canonical(url))

    def __contains__(self, url: str) -> bool:
        return canonical(url) in self.urls

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.urls), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", action="append", default=[], help="a page to read")
    ap.add_argument("--dir", help="a folder of saved .html or .txt notices")
    ap.add_argument("--out", default=str(STATE / "proposals.json"))
    ap.add_argument("--again", action="store_true", help="re-read sources already seen")
    args = ap.parse_args()

    seen = Seen(STATE / "seen.json")
    proposals, rejected = [], []

    sources: list[tuple[str, str | None]] = [(u, None) for u in args.url]
    if args.dir:
        for f in sorted(Path(args.dir).iterdir()):
            if f.suffix.lower() in (".html", ".htm", ".txt"):
                sources.append((str(f), f.read_text(errors="replace")))

    if not sources:
        ap.error("give --url or --dir")

    for source, body in sources:
        if not args.again and source in seen:
            print(f"skip (seen)  {source}")
            continue
        try:
            if body is None:
                body = fetch(source)
            text = strip_html(body) if "<" in body[:2000] else body
            p = propose(text, source)
        except Exception as e:
            rejected.append((source, repr(e)))
            print(f"failed       {source}: {e}")
            continue

        issues = problems(p)
        seen.add(source)
        if issues:
            rejected.append((source, "; ".join(issues)))
            print(f"rejected     {p.name[:50]}: {issues[0]}")
            continue

        proposals.append(asdict(p))
        gaps = f", {len(p.unanswered)} unanswered" if p.unanswered else ""
        print(f"proposed     {p.name[:50]}{gaps}")

    seen.save()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proposals, indent=2, ensure_ascii=False))

    print(f"\n{len(proposals)} proposals -> {out}")
    if rejected:
        print(f"{len(rejected)} not proposed:")
        for src, why in rejected:
            print(f"  {src[:60]}: {why[:90]}")


if __name__ == "__main__":
    main()
