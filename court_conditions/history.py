"""
history.py — "how many playable hours does a typical Reno March have?"

WHAT THIS ANSWERS
The forecast tells you about this week. This tells you about the season. If
you're planning when to run clinics, or arguing that the team needs indoor
time in February, the useful number is seasonal: "February averages 62
playable hours a month and June averages 310."

HOW IT WORKS
We pull several years of real hourly weather from Open-Meteo's archive, run
every single one of those hours through the exact same scoring rules the
forecast uses, and count how many cleared the playable threshold.

DESIGN CHOICE: reusing the live scoring rules on historical data, rather than
writing a separate simpler "was it nice" check, is what makes this view
trustworthy. It means the history and the forecast can never disagree about
what "playable" means — and when you tune a weight in scoring.yaml, the
historical chart updates to match. You can literally watch your tuning change
the past, which is a good sanity check on whether a weight is sensible.

A NOTE ON COST
Three years of hourly data is about 26,000 rows per court, and each row runs
five rules. That's the slowest thing this app does — a few seconds per court.
It's cached for 30 days (see cache.py), so you pay it once.
"""

from __future__ import annotations

import calendar

import pandas as pd

from .config import Court, load_settings
from .daylight import playable_light_window
from .scoring import playable_threshold, score_series
from .weather import fetch_history_years

MONTH_NAMES = list(calendar.month_abbr)  # ['', 'Jan', 'Feb', ...] — index 1-12

# Reno's tennis seasons, grouped for the seasonal rollup. Meteorological
# seasons (Dec-Feb winter) rather than astronomical ones, because they line up
# with how weather actually behaves.
SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall": [9, 10, 11],
}


def season_of(month: int) -> str:
    """Which season a month number belongs to."""
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "Unknown"


def monthly_playable_hours(court: Court, years: int | None = None) -> pd.DataFrame:
    """
    Count playable hours per calendar month over the past few years.

    Returns one row per (year, month) with:
        year, month, month_name, season
        playable_hours    hours scoring at or above the threshold
        daylight_hours    hours of natural light in that month
        lit_hours         hours with usable light: daylight OR working lights
        total_hours       hours in the month with weather data
        playable_pct      playable as a share of LIT hours
        avg_score         mean playability across the whole month

    `playable_pct` is measured against LIT hours. Dividing by all 24 would
    make every winter month look terrible for a reason that has nothing to do
    with weather — December's problem is darkness. Comparing against the hours
    you could possibly have played separates "the weather was bad" from "there
    was no light", which are different problems with different solutions (one
    needs an indoor court, the other needs lights).

    A NOTE ON WHY THE DENOMINATOR IS `lit_hours`:
    the first version of this used daylight hours, and lit courts came out at
    133% playable, because hours played under floodlights landed in the
    numerator while the denominator counted daylight alone. A percentage over
    100 is always a sign that the top and bottom are counting different
    things. The denominator has to be every hour you could have played,
    lights included.
    """
    years = years or load_settings().get("data", {}).get("history_years", 3)

    weather = fetch_history_years(court, years)
    if weather.empty:
        return pd.DataFrame()

    scored = score_series(court, weather)
    if scored.empty:
        return pd.DataFrame()

    threshold = playable_threshold()

    scored = scored.copy()
    scored["year"] = scored["time"].dt.year
    scored["month"] = scored["time"].dt.month
    scored["is_playable"] = scored["score"] >= threshold
    # `light` comes from the scoring pass: "daylight", "lights", or "dark".
    scored["is_daylight"] = scored["light"] == "daylight"
    scored["is_lit"] = scored["light"] != "dark"

    grouped = (
        scored.groupby(["year", "month"])
        .agg(
            playable_hours=("is_playable", "sum"),
            daylight_hours=("is_daylight", "sum"),
            lit_hours=("is_lit", "sum"),
            total_hours=("score", "size"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )

    # Drop partial months at either end of the window. A month with only 8
    # days of data would show a misleadingly low hour count next to full
    # months, purely as an artifact of where the fetch window happened to end.
    grouped = grouped[grouped["total_hours"] >= 24 * 25]

    if grouped.empty:
        return pd.DataFrame()

    grouped["month_name"] = grouped["month"].map(lambda m: MONTH_NAMES[m])
    grouped["season"] = grouped["month"].map(season_of)
    # Guard against dividing by zero — possible in theory near the poles,
    # never in Reno, but a crash in a chart is worse than a zero.
    grouped["playable_pct"] = (
        grouped["playable_hours"] / grouped["lit_hours"].replace(0, pd.NA) * 100
    ).fillna(0.0)
    grouped["avg_score"] = grouped["avg_score"].round(1)

    return grouped.sort_values(["year", "month"]).reset_index(drop=True)


def monthly_averages(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-year table into one average row per calendar month.

    This is the "what is a typical March here" view. Averaging across years
    smooths out the fact that one particular March happened to be a washout.
    `years_counted` is carried through so the chart can be honest about how
    much data each average rests on — an average of two years is a weaker
    claim than an average of five.
    """
    if monthly.empty:
        return pd.DataFrame()

    averages = (
        monthly.groupby("month")
        .agg(
            playable_hours=("playable_hours", "mean"),
            daylight_hours=("daylight_hours", "mean"),
            lit_hours=("lit_hours", "mean"),
            playable_pct=("playable_pct", "mean"),
            avg_score=("avg_score", "mean"),
            years_counted=("year", "nunique"),
        )
        .reset_index()
    )

    averages["month_name"] = averages["month"].map(lambda m: MONTH_NAMES[m])
    averages["season"] = averages["month"].map(season_of)
    averages["playable_hours"] = averages["playable_hours"].round(0)
    averages["playable_pct"] = averages["playable_pct"].round(1)
    averages["avg_score"] = averages["avg_score"].round(1)

    return averages.sort_values("month").reset_index(drop=True)


def seasonal_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """Average playable hours per month, grouped into the four seasons."""
    if monthly.empty:
        return pd.DataFrame()

    summary = (
        monthly.groupby("season")
        .agg(
            avg_playable_hours_per_month=("playable_hours", "mean"),
            avg_score=("avg_score", "mean"),
        )
        .reset_index()
    )
    summary["avg_playable_hours_per_month"] = summary[
        "avg_playable_hours_per_month"
    ].round(0)
    summary["avg_score"] = summary["avg_score"].round(1)

    # Order the seasons the way a year runs, overriding alphabetical order.
    order = ["Winter", "Spring", "Summer", "Fall"]
    summary["_sort"] = summary["season"].map(order.index)
    return summary.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
