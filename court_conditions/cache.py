"""
cache.py — stops the app from re-asking Open-Meteo for data it already has.

WHY THIS FILE EXISTS
Streamlit re-runs your entire script from the top every time you click
anything. Without caching, picking a different court from a dropdown would
fire a fresh HTTP request for every court on screen. Open-Meteo is free and
generous, and in return they ask for reasonable use — hammering a free public
API that a lot of people depend on is bad manners. Caching is the polite thing
to do, and it also makes the dashboard feel instant.

HOW IT WORKS
`requests_cache` wraps the normal `requests` library. You use the session
exactly like `requests.get(...)`, but the first call saves the response into a
small SQLite file on disk and later identical calls read from that file
instead of the network. Because it's on disk (not in memory), the cache also
survives restarting the app — so quitting and relaunching doesn't re-download
three years of weather.

TWO CACHES, TWO VERY DIFFERENT LIFETIMES
This is the main design decision in this file:

  * The FORECAST is a prediction about the future. It genuinely changes — the
    weather service updates it several times a day. Cached for 30 minutes.

  * The ARCHIVE is what the weather actually was, years ago. It will never
    change. Cached for 30 days, and honestly could be cached forever.

Using one shared lifetime would mean either re-downloading historical data
needlessly or showing you a stale forecast. Splitting them costs a few lines
and gets both right.
"""

from __future__ import annotations

from pathlib import Path

import requests_cache

from .config import PROJECT_ROOT, load_settings

# Cache files live in data/, which .gitignore excludes. They are derived data —
# anyone who clones the repo can regenerate them by running the app, so there
# is no reason to commit megabytes of weather to git.
CACHE_DIR = PROJECT_ROOT / "data"


def _make_session(name: str, expire_after: int) -> requests_cache.CachedSession:
    """Build one cached session writing to data/<name>.sqlite."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return requests_cache.CachedSession(
        cache_name=str(CACHE_DIR / name),
        backend="sqlite",
        expire_after=expire_after,
        # Only cache successful responses. Without this, a momentary 500 from
        # the API would get cached and the app would keep showing that error
        # for the next 30 minutes even after the service recovered.
        allowable_codes=(200,),
        # Identify ourselves. Good API manners: if this app ever misbehaved,
        # Open-Meteo could see what it was instead of an anonymous script.
        headers={"User-Agent": "court-conditions/1.0 (personal tennis project)"},
    )


# These are module-level singletons — created once when the module is first
# imported, then reused. Two sessions would otherwise mean two SQLite
# connections fighting over the same file.
_forecast_session: requests_cache.CachedSession | None = None
_archive_session: requests_cache.CachedSession | None = None


def forecast_session() -> requests_cache.CachedSession:
    """Session for the 7-day forecast (short cache — forecasts change)."""
    global _forecast_session
    if _forecast_session is None:
        seconds = load_settings().get("data", {}).get("forecast_cache_seconds", 1800)
        _forecast_session = _make_session("forecast_cache", seconds)
    return _forecast_session


def archive_session() -> requests_cache.CachedSession:
    """Session for historical weather (long cache — the past is fixed)."""
    global _archive_session
    if _archive_session is None:
        seconds = load_settings().get("data", {}).get("archive_cache_seconds", 2592000)
        _archive_session = _make_session("archive_cache", seconds)
    return _archive_session


def cache_status() -> dict[str, str]:
    """
    Human-readable summary of what's cached, shown in the dashboard footer.

    Useful when something looks wrong: if the forecast seems stale, this tells
    you whether you're looking at fresh data or a cached copy.
    """
    status: dict[str, str] = {}
    for label, path in (
        ("forecast", CACHE_DIR / "forecast_cache.sqlite"),
        ("history", CACHE_DIR / "archive_cache.sqlite"),
    ):
        if path.exists():
            size_kb = path.stat().st_size / 1024
            status[label] = f"{size_kb:,.0f} KB cached"
        else:
            status[label] = "empty (nothing fetched yet)"
    return status


def clear_cache() -> None:
    """Delete both cache files. The dashboard exposes this as a button."""
    for session in (forecast_session(), archive_session()):
        session.cache.clear()


def cache_files() -> list[Path]:
    """Paths of the cache files on disk, for the README and debugging."""
    return [CACHE_DIR / "forecast_cache.sqlite", CACHE_DIR / "archive_cache.sqlite"]
