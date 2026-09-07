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
