"""
scrape.py — optionally check public parks pages for published court schedules.

READ THIS SECTION BEFORE RUNNING ANYTHING HERE.

Scraping is the part of this project with actual ethical weight, so the code
itself enforces the rules, leaving nothing to whoever happens to run it:

  1. ROBOTS.TXT IS CHECKED FIRST, AND IT IS BINDING.
     Before any page is fetched, we read that site's robots.txt and ask
     whether our user agent is allowed to fetch that specific URL. If the
     answer is no, we skip it and log why. There is no override flag. A
     robots.txt is a site operator telling you what they want, and that
     request deserves respect on its own terms, advisory or otherwise.

  2. ONE REQUEST EVERY 3 SECONDS, MINIMUM.
     Enforced by a sleep in the fetch function, so it cannot be forgotten.
     These are small city and parks department servers. A scraper that fires
     hundreds of requests a second is indistinguishable from an attack, and
     can genuinely knock a small site over.

  3. NOTHING BEHIND A LOGIN. EVER.
     There are no credentials in this file and no session handling. If a
     schedule requires an account to view, it is not public data and this
     project does not touch it. Automating past a login also usually breaks
     the site's terms of service, which is a different and more serious
     problem than robots.txt.

  4. EVERY SKIP IS LOGGED WITH A REASON.
     The output tells you what it fetched, what it skipped, and why. A scraper
     that silently returns nothing is impossible to trust or debug.

  5. IT IDENTIFIES ITSELF HONESTLY.
     The User-Agent says what this is. If a site operator sees this traffic in
     their logs and wants it to stop, they can tell who it is and block it.

WHAT THIS ACTUALLY FINDS
Honestly: probably not much that's structured. Most parks departments publish
court availability as a PDF, inside a reservation system, or leave it offline
entirely. This module is built to look, report clearly on what it found, and
fail gracefully when the answer is "nothing usable" — the most likely outcome,
and a perfectly fine one. The app works entirely without it; scraped data sits
on top of your hand-written courts.yaml as an optional bonus layer.

Run it with:  python scripts/check_schedules.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
import urllib.parse
import urllib.robotparser
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Identify ourselves honestly. A real contact route matters: change the URL to
# your own GitHub repo so anyone who sees this traffic can find out what it is.
USER_AGENT = (
    "court-conditions-bot/1.0 "
    "(personal high-school tennis project; "
    "https://github.com/sahishnu-m/court-conditions)"
)

# Minimum gap between requests to the same host, in seconds. The brief said
# 2-3 seconds; we take the slow end of that range as the default because being
# slower than asked costs us nothing and costs them less.
RATE_LIMIT_SECONDS = 3.0

# Where the run's findings get written.
OUTPUT_PATH = PROJECT_ROOT / "data" / "scrape_report.json"

# Public parks pages worth checking. Treat these as starting points: city
# websites get reorganised constantly and these URLs may 404 by the time you
# read this. A 404 is logged like any other skip and the run carries on.
CANDIDATE_SOURCES = [
    {
        "name": "City of Reno — Tennis",
        "url": "https://www.reno.gov/government/departments/parks-recreation-community-services/athletics/tennis",
        "looking_for": "tennis programs, Reno Tennis Center hours, league info",
    },
    {
        "name": "City of Reno — Parks & Facilities Directory",
        "url": "https://www.reno.gov/government/departments/parks-recreation-community-services/parks-facilities-directory",
        "looking_for": "per-park amenity listings including court counts",
    },
    {
        "name": "City of Sparks — Find a Park or Facility",
        "url": "https://www.cityofsparks.us/rec_home/parks___facilities/find_a_park_or_facility.php",
        "looking_for": "Sparks park amenities and court counts",
    },
    {
        "name": "Washoe County Regional Parks",
        "url": "https://www.washoecounty.gov/parks/",
        "looking_for": "regional park facilities outside the two city systems",
    },
]


@dataclass
class SourceResult:
    """What happened when we looked at one source."""

    name: str
    url: str
    status: str  # "fetched" | "skipped" | "error"
    reason: str  # always populated, especially for skips
    http_status: int | None = None
    bytes_received: int | None = None
    mentions_tennis: bool | None = None
    checked_at: str = ""


class PoliteFetcher:
    """
    A requests wrapper that cannot be used impolitely.

    DESIGN CHOICE: the rate limit and robots.txt check live INSIDE the fetch
    method, which takes them off the caller's plate entirely. Left as the
    caller's job, some future version of this code would forget one — that's
    how every badly-behaved scraper gets written. Making good behaviour the
    only available path is more reliable than remembering to be good.
    """

    def __init__(self, rate_limit_seconds: float = RATE_LIMIT_SECONDS) -> None:
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_at: float = 0.0
        # Cache robots.txt per host so we fetch each one only once per run.
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _wait_turn(self) -> None:
        """Sleep as long as needed to honour the rate limit."""
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.rate_limit_seconds - elapsed
        if remaining > 0:
            logger.debug("Rate limit: sleeping %.1fs", remaining)
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        """
        Fetch and parse a host's robots.txt, caching the result.

        Returns None if robots.txt could not be read at all. How we treat that
        case is a real decision — see `allowed()`.
        """
        parts = urllib.parse.urlparse(url)
        host_key = f"{parts.scheme}://{parts.netloc}"

        if host_key in self._robots:
            return self._robots[host_key]

        robots_url = f"{host_key}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)

        self._wait_turn()  # robots.txt is a request too, and counts
        try:
            response = self._session.get(robots_url, timeout=15)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                logger.info("Read robots.txt from %s", robots_url)
                self._robots[host_key] = parser
            elif response.status_code in (401, 403):
                # A protected robots.txt means "you are not welcome here".
                logger.warning("robots.txt at %s is protected — treating site as disallowed", robots_url)
                self._robots[host_key] = None
            else:
                # 404 means no robots.txt exists. By the standard, that means
                # everything is permitted. We still rate-limit.
                logger.info("No robots.txt at %s (HTTP %s) — nothing disallowed", robots_url, response.status_code)
                parser.parse([])  # empty ruleset = allow all
                self._robots[host_key] = parser
        except requests.exceptions.RequestException as exc:
            logger.warning("Could not fetch %s: %s", robots_url, exc)
            self._robots[host_key] = None

        return self._robots[host_key]

    def allowed(self, url: str) -> tuple[bool, str]:
        """
        May we fetch this URL? Returns (allowed, reason).

        DESIGN CHOICE: if robots.txt can't be read, we refuse.

        The permissive alternative — "couldn't check, so go ahead" — is what
        most scrapers do, and it gets the burden of proof backwards.
        Confirming permission is what grants it. Failing closed costs us a
        page we might have been allowed to fetch; failing open risks hammering
        a site that explicitly asked us to stay away.
        """
        parser = self._robots_for(url)
        if parser is None:
            return False, "could not read robots.txt — failing closed (assuming not allowed)"

        if parser.can_fetch(USER_AGENT, url):
            return True, "allowed by robots.txt"
        return False, "disallowed by robots.txt"

    def fetch(self, url: str) -> requests.Response:
        """Fetch a URL, rate-limited. Call `allowed()` first."""
        self._wait_turn()
        logger.info("Fetching %s", url)
        return self._session.get(url, timeout=20)


def check_sources(sources: list[dict] | None = None) -> list[SourceResult]:
    """
    Check each candidate source and report what we found.

    This deliberately stops short of parsing schedules into structured data.
    Every one of these sites has a different layout, and a parser written
    against today's HTML would silently break the next time a city redesigns
    its website — silently producing wrong court hours is worse than producing
    none. So we verify the page is reachable and permitted, note whether it
    even mentions tennis, and leave the reading to you.

    If you find a source that publishes a genuinely structured schedule (an
    iCal feed, a JSON endpoint, a CSV), that's the point at which writing a
    real parser becomes worthwhile — a stable format is safe to parse.
    """
    sources = sources or CANDIDATE_SOURCES
    fetcher = PoliteFetcher()
    results: list[SourceResult] = []
    now = dt.datetime.now().isoformat(timespec="seconds")

    for source in sources:
        url = source["url"]
        name = source["name"]

        is_allowed, reason = fetcher.allowed(url)
        if not is_allowed:
            logger.warning("SKIP %s — %s", name, reason)
            results.append(
                SourceResult(
                    name=name, url=url, status="skipped", reason=reason, checked_at=now
                )
            )
            continue

        try:
            response = fetcher.fetch(url)
        except requests.exceptions.RequestException as exc:
            logger.error("ERROR %s — %s", name, exc)
            results.append(
                SourceResult(
                    name=name,
                    url=url,
                    status="error",
                    reason=f"request failed: {exc}",
                    checked_at=now,
                )
            )
            continue

        if response.status_code != 200:
            # A 404 or 403 on a URL that clearly works in a browser usually
            # means the site is refusing automated clients, either through a
            # CDN bot filter or by serving different content based on the
            # User-Agent.
            #
            # WE DO NOT WORK AROUND THIS. Disguising the scraper as Chrome
            # would very likely get these pages, but a site that blocks bots
            # is expressing the same preference as a robots.txt Disallow,
            # just through a different mechanism. Honouring robots.txt while
            # spoofing a browser to defeat bot detection would follow the
            # letter of the rule while breaking its intent.
            #
            # The correct response is the one below: record it plainly and
            # go read the page in a browser yourself.
            hint = ""
            if response.status_code in (403, 404, 406, 429):
                hint = (
                    " — this URL may work fine in a browser; a city site "
                    "refusing our identified bot is a signal to read it by "
                    "hand rather than to disguise the scraper"
                )
            results.append(
                SourceResult(
                    name=name,
                    url=url,
                    status="error",
                    reason=f"HTTP {response.status_code}{hint}",
                    http_status=response.status_code,
                    checked_at=now,
                )
            )
            continue

        body = response.text.lower()
        mentions_tennis = "tennis" in body

        results.append(
            SourceResult(
                name=name,
                url=url,
                status="fetched",
                reason=(
                    "page fetched; mentions tennis — worth reading by hand"
                    if mentions_tennis
                    else "page fetched but no mention of tennis; probably not useful"
                ),
                http_status=200,
                bytes_received=len(response.content),
                mentions_tennis=mentions_tennis,
                checked_at=now,
            )
        )

    return results


def write_report(results: list[SourceResult]) -> Path:
    """Save the run's findings to data/scrape_report.json."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "user_agent": USER_AGENT,
        "rate_limit_seconds": RATE_LIMIT_SECONDS,
        "results": [asdict(r) for r in results],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return OUTPUT_PATH


def load_report() -> dict | None:
    """Read the last report back, for the dashboard to display. None if never run."""
    if not OUTPUT_PATH.exists():
        return None
    try:
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
