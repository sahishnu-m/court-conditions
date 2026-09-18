"""
Check public parks pages for published court schedules.

Run from the repo root:   python scripts/check_schedules.py

This lives in a SEPARATE SCRIPT, kept well outside the dashboard, on purpose.
Scraping touches other people's servers, so it should only ever happen because
you explicitly decided to run it. Were this wired into app.py, every visitor to
your deployed dashboard would quietly fire requests at the City of Reno's
website on your behalf.

Read the header of court_conditions/scrape.py for the full set of rules this
follows (robots.txt is binding, 3 seconds between requests, nothing behind a
login, every skip logged).

Expect this to be slow — that's the rate limit working as intended.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running this file directly from the repo root without installing the
# package: add the project root to Python's import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from court_conditions.scrape import (  # noqa: E402  (import after path setup)
    CANDIDATE_SOURCES,
    RATE_LIMIT_SECONDS,
    check_sources,
    write_report,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Checking public parks pages for court schedule information.")
    print(f"Rate limit: one request every {RATE_LIMIT_SECONDS} seconds.")
    print("robots.txt is checked first for every source and is binding.\n")

    results = check_sources(CANDIDATE_SOURCES)

    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)

    for result in results:
        icon = {"fetched": "[OK]  ", "skipped": "[SKIP]", "error": "[ERR] "}.get(
            result.status, "[?]   "
        )
        print(f"{icon} {result.name}")
        print(f"        {result.url}")
        print(f"        {result.reason}")
        if result.bytes_received:
            print(f"        {result.bytes_received:,} bytes")
        print()

    path = write_report(results)
    print(f"Report written to {path}")
    print("The dashboard shows this report at the bottom of the page.\n")

    fetched = sum(1 for r in results if r.status == "fetched")
    tennis = sum(1 for r in results if r.mentions_tennis)
    print(f"{fetched} of {len(results)} sources fetched; {tennis} mention tennis.")
    print(
        "\nNOTE: this intentionally does not parse schedules into data. City\n"
        "sites get redesigned and an HTML parser would silently start\n"
        "producing wrong court hours. Open the pages that mention tennis and\n"
        "copy anything useful into config/courts.yaml by hand."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
