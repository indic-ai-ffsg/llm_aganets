"""The discovery service: three stages over HTTP, and no state.

    uvicorn service:app --host 0.0.0.0 --port 8080

Why it holds nothing
--------------------
A run lasts about half an hour and has to survive a restart, a redeploy and two
instances. That is a job for the queue and the schema the platform already has
— `backend/internal/platform/redisx` drains jobs with per-consumer processing
lists and stranded-job recovery, and the tables live in Postgres behind RLS.
None of that should be reimplemented here in a language that cannot see either.

So this service is a pure function of its request. It knows how to search, to
judge and to read; it does not know what a run is, which URLs have been seen,
or where a draft goes. The backend owns all of that, calls these three
endpoints one page at a time, and writes the drafts itself through the registry
service — which means every listing discovery creates passes the same
validation, the same audit trail and the same row-level policies as one typed
in by hand.

The old portal did it the other way: the pipeline held the run, inserted
straight into the scholarships table, and chained itself over HTTP because a
Vercel function dies at 300 seconds. That chain broke whenever a query took
longer than the ceiling, which is why its dialog has a Resume button. There is
no ceiling here and nothing to resume.

Authentication
--------------
One shared secret in `DISCOVERY_TOKEN`, required on every route but /healthz.
The service holds a Gemini key and will spend it for anyone who can reach it,
and a Railway URL is not a secret — it is in an environment variable in another
service and in whatever shell last read it.
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosscheck  # noqa: E402
import llm  # noqa: E402

log = logging.getLogger("discovery")

app = FastAPI(
    title="Indic AI scholarship discovery",
    description=__doc__,
    docs_url="/docs" if os.environ.get("ENV") != "production" else None,
)


# --- authentication ----------------------------------------------------------

def require_token(x_discovery_token: str = Header(default="")) -> None:
    """A constant-time check against DISCOVERY_TOKEN.

    Refuses to run unauthenticated rather than defaulting open: an accidentally
    public instance of this is somebody else's Gemini bill and a search tool
    pointed wherever they like.
    """
    expected = os.environ.get("DISCOVERY_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            503,
            "DISCOVERY_TOKEN is not set on this service, so it will not serve "
            "requests. Set it here and in the API's DISCOVERY_TOKEN.")
    if not hmac.compare_digest(x_discovery_token.strip(), expected):
        raise HTTPException(401, "Bad or missing X-Discovery-Token.")


# --- seed queries ------------------------------------------------------------

def seed_queries() -> list[str]:
    """The fifteen the old portal shipped, with the year computed.

    Kept verbatim because they have a track record: the portal's
    discovery_queries table recorded hits, valid pages and drafts per query
    across every run, and these are the ones that produced drafts. The year is
    computed rather than written so a query does not quietly go stale in
    January.
    """
    year = date.today().year
    nxt = year + 1
    return [
        f"scholarship for students with disabilities India {year}",
        f"scholarship disabled students India {year} {nxt} apply online",
        "scholarship for physically disabled students India education",
        "scholarship for visually impaired students India education",
        "scholarship for hearing impaired deaf students India education",
        "scholarship for students with intellectual disability India",
        "government scholarship students with disabilities India RPWD",
        "post matric scholarship disabled students state government India",
        "top class education scholarship students with disabilities",
        "national scholarship portal disabled students India",
        "private corporate scholarship students with disabilities India",
        "CSR scholarship disability India HDFC SBI Bajaj education",
        "scholarship cerebral palsy autism students India education",
        "NGO scholarship physically challenged students India higher education",
        "fellowship grant students with disabilities India postgraduate",
    ]


# --- request bodies ----------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    # An operator's own key, when the service's has expired or hit quota. Never
    # logged and never stored — it lives for the length of the call.
    api_key: str | None = None


class ClassifyRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    api_key: str | None = None


class ExtractRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    # From the backend, which reads it out of listing_tag. The vocabulary is
    # closed and is not an enum, so this service cannot know it.
    allowed_tags: list[str] = Field(default_factory=list, max_length=200)
    api_key: str | None = None
    # The independent read costs one HTTP GET. Off is available for a page that
    # blocks the fetch but that the model can reach through its own tool.
    verify: bool = True


class TranslateRequest(BaseModel):
    # field name -> the English text. Named by the caller rather than fixed
    # here, because which fields have somewhere to be stored is the backend's
    # question: today it is summary and description, and nothing in this
    # service should have to change when that grows.
    fields: dict[str, str] = Field(default_factory=dict, max_length=40)
    languages: list[str] = Field(default_factory=lambda: ["hi"], max_length=8)
    # Per-field character ceiling, from the backend, which is the side that
    # knows the column widths. A translation over its ceiling is cut here and
    # reported — the alternative is the registry refusing the whole draft, and
    # losing a scholarship over an optional translation is the wrong trade.
    limits: dict[str, int] = Field(default_factory=dict, max_length=40)
    api_key: str | None = None


# --- settling the deadline ---------------------------------------------------

def settle_closing_date(proposal: llm.Proposal, check: crosscheck.Check,
                        today: date | None = None) -> None:
    """Decide the closing date where the two readers disagree, and say so.

    The two are not equal evidence. The pattern reader's date arrived with the
    sentence it was read from; the model's arrived with nothing, and the way it
    goes wrong is specific — the right day and month, and a year from whenever it
    last saw the page.

    That failure is silent and it is expensive. The public directory selects
    `closes_at > now()` (migration 0043), so a stale year does not show a student
    that the scheme has closed. It removes the scheme from the site, while the
    panel goes on reporting it as Published.

    Two cases are settled here and no others:

        the model gave none      the evidenced date is adopted
        the model's has passed
          and the page's has not the evidenced date wins

    Everything else — two future dates that differ, two past ones — is left as
    the model wrote it and annotated by crosscheck. That is the line this
    service holds everywhere else: which reader is right about a page is a
    judgement, and the operator reviewing the draft is the one making it. These
    two are not judgements. One is a value against no value, and the other is
    a comparison against today.
    """
    today = today or date.today()
    read = check.closes_at
    if not read:
        return

    if not proposal.closes_at:
        proposal.closes_at = read
        proposal.issues.append(
            f"closing date {read} was read off the page by the pattern reader; the "
            "model proposed none"
            + (", and the notice gave no year, so it was taken as the next one "
               "coming — check it" if check.closes_at_inferred else ""))
    elif proposal.closes_at != read and proposal.closes_at < today.isoformat() <= read:
        proposal.issues.append(
            f"closing date: the model said {proposal.closes_at}, which has passed, and "
            f"the page reads {read}. Using {read} — the model's date would have taken "
            "the scheme out of the directory rather than showing it as closed.")
        proposal.closes_at = read
    else:
        return

    # The date just adopted may have crossed the opening date the model gave,
    # and the API refuses a window that closes before it opens. The evidenced
    # date stays and the unevidenced one goes, which is the same ordering of
    # trust that got us here.
    if proposal.opens_at and proposal.closes_at <= proposal.opens_at:
        proposal.issues.append(
            f"opens_at {proposal.opens_at} is not before the closing date "
            f"{proposal.closes_at} that was read off the page, so it was dropped")
        proposal.opens_at = None


# --- routes ------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    """Liveness, and whether this instance could actually do any work.

    `key_configured` is a boolean and not a masked key: a mask is still four
    characters of a live credential in a log line that nobody treats as
    sensitive because it came from a health check.
    """
    return {
        "ok": True,
        "model": llm.MODEL,
        "key_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "token_configured": bool(os.environ.get("DISCOVERY_TOKEN", "").strip()),
        "max_urls_per_query": llm.MAX_URLS_PER_QUERY,
    }


@app.get("/queries", dependencies=[Depends(require_token)])
def queries() -> dict:
    return {"queries": seed_queries()}


@app.post("/search", dependencies=[Depends(require_token)])
def do_search(req: SearchRequest) -> dict:
    try:
        urls = llm.search(req.query, req.api_key)
    except llm.ModelError as e:
        # 502, not 500: this service is fine and the upstream model is not, and
        # the backend retries a 502 rather than failing the run.
        raise HTTPException(502, str(e)) from e
    return {"query": req.query, "urls": urls}


@app.post("/classify", dependencies=[Depends(require_token)])
def do_classify(req: ClassifyRequest) -> dict:
    try:
        verdict = llm.classify(req.url, req.api_key)
    except llm.ModelError as e:
        raise HTTPException(502, str(e)) from e
    return {"url": req.url, **asdict(verdict)}


@app.post("/translate", dependencies=[Depends(require_token)])
def do_translate(req: TranslateRequest) -> dict:
    """The student-facing prose, in each language asked for.

    Separate from /extract on purpose, and the whole reason is the failure
    mode: a translation that comes back unusable should cost the translation
    and not the listing. See llm.translate.
    """
    try:
        translations, issues = llm.translate(
            req.fields, req.languages, req.limits, req.api_key)
    except llm.ModelError as e:
        raise HTTPException(502, str(e)) from e
    return {"translations": translations, "issues": issues}


@app.post("/extract", dependencies=[Depends(require_token)])
def do_extract(req: ExtractRequest) -> dict:
    try:
        proposal = llm.extract(req.url, req.allowed_tags, req.api_key)
    except llm.ModelError as e:
        raise HTTPException(502, str(e)) from e

    evidence: dict[str, str] = {}
    verified = False

    if req.verify:
        check = crosscheck.against(req.url, proposal.rules, proposal.closes_at)
        evidence = check.evidence
        verified = check.ran
        proposal.issues.extend(check.notes)
        # After the notes, so the sentence explaining a swap reads after the one
        # reporting the disagreement that caused it.
        settle_closing_date(proposal, check)

    body = asdict(proposal)
    issues = body.pop("issues")
    return {
        "url": req.url,
        "proposal": body,
        # Separate from the proposal because they are not fields to import —
        # they are what an operator has to look at before publishing, and the
        # backend stores them on the draft for exactly that.
        "issues": issues,
        # field name -> the sentence a pattern read the value from, so the
        # panel can show the evidence beside the number.
        "evidence": evidence,
        "independently_verified": verified,
    }
