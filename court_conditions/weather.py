"""
weather.py: everything that talks to Open-Meteo.

WHY OPEN-METEO
It's free, needs no API key, and serves both forecast and multi-year history
from the same query format. No key matters more than it sounds: a key would
have to be kept out of git, injected as a secret on Streamlit Cloud, and
rotated if leaked. Choosing a keyless API removes that entire problem, which
is a real design consideration for a project you want to deploy publicly.

TWO ENDPOINTS
  api.open-meteo.com/v1/forecast, now through +7 days
  archive-api.open-meteo.com/v1/archive, historical reanalysis data

The archive endpoint lags real time by about 5 days, which is why the
historical functions here never ask for the last week.

UNITS
We ask Open-Meteo for Fahrenheit, mph, and inches directly, using their
unit parameters. The alternative, fetching metric and converting, would
mean conversion math scattered through the code and a permanent risk of a
unit bug. Pushing the conversion to the API is free and eliminates that.

EVERY FUNCTION HERE RETURNS A pandas DataFrame with these exact columns:
    time            (timezone-aware datetime, Reno local)
    temp_f          (float, Fahrenheit)
    precip_in       (float, inches of precipitation in that hour)
    wind_mph        (float, sustained wind speed)
    wind_gust_mph   (float, gust speed)
Keeping one shape for both forecast and history means scoring.py doesn't need
to know or care which one it was handed.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from .cache import archive_session, forecast_session
from .config import Court, get_timezone, load_settings

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# The hourly variables we need, in Open-Meteo's naming.
HOURLY_VARIABLES = [
    "temperature_2m",  # air temp at 2 metres, standard "how warm is it"
    "precipitation",  # rain + snow melt for that hour
    "wind_speed_10m",  # sustained wind at 10 metres
    "wind_gusts_10m",  # peak gust, matters more than average for tennis
]

# Renaming map from Open-Meteo's column names to ours. Doing this in one place
# means if they ever rename a field, this dict is the only thing to fix.
COLUMN_RENAMES = {
    "temperature_2m": "temp_f",
    "precipitation": "precip_in",
    "wind_speed_10m": "wind_mph",
    "wind_gusts_10m": "wind_gust_mph",
}

# The archive dataset is finalised on a delay, so asking for yesterday returns
# nulls. Stopping this many days back avoids a ragged edge in the charts.
ARCHIVE_LAG_DAYS = 6


class WeatherUnavailable(RuntimeError):
    """
    Raised when we genuinely cannot get weather.

    A named exception (rather than letting a requests error bubble up) lets
    the dashboard catch exactly this and show 'the weather service is down'
    instead of a stack trace the user can't act on.
    """


def _common_params(court: Court) -> dict[str, object]:
    """Query parameters shared by both endpoints."""
    return {
        "latitude": court.lat,
        "longitude": court.lon,
        "hourly": ",".join(HOURLY_VARIABLES),
        # Ask for US units so they match scoring.yaml with no conversion.
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        # Getting timestamps already in Reno time means every downstream
        # comparison ("is this hour after sunset?") is a plain comparison,
        # with no UTC offset arithmetic to get wrong.
        "timezone": get_timezone(),
    }


def _to_frame(payload: dict) -> pd.DataFrame:
    """
    Convert Open-Meteo's JSON into our standard DataFrame.

    Open-Meteo returns parallel arrays, one list of timestamps and one list
    per variable, all the same length, rather than a list of row objects.
    That maps directly onto a DataFrame constructor.
    """
    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        raise WeatherUnavailable("Open-Meteo returned no hourly data.")

    frame = pd.DataFrame(hourly)
    # Timestamps arrive as ISO strings like "2026-09-17T14:00". They are
    # already in Reno local time (we asked for that), so we attach the
    # timezone with tz_localize. Using tz_convert would shift the clock times.
    #
    # DAYLIGHT SAVING TIME, the two arguments below are load-bearing.
    # Twice a year local time does something impossible:
    #
    #   "Fall back" (early November): 1:00am–1:59am happens TWICE. A bare
    #   timestamp of 01:30 is AMBIGUOUS, pandas can't tell which one you mean
    #   and raises an error rather than guessing. `ambiguous=True` says "treat
    #   it as the first pass, the one still on daylight time".
    #
    #   "Spring forward" (March): 2:00am–2:59am NEVER HAPPENS. Those
    #   timestamps are NONEXISTENT. `nonexistent="shift_forward"` moves them
    #   to 3:00am instead of erroring.
    #
    # This never comes up in a 7-day forecast, so it's invisible until you
    # pull multiple years of history and hit a November, which is exactly
    # how this bug was found. Either choice affects one hour a year at 1am,
    # when nobody is playing tennis, so what matters here is simply that the
    # app keeps running.
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize(
        get_timezone(), ambiguous=True, nonexistent="shift_forward"
    )
    frame = frame.rename(columns=COLUMN_RENAMES)

    for column in COLUMN_RENAMES.values():
        if column not in frame.columns:
            frame[column] = float("nan")

    # Open-Meteo uses null for "no data for this hour". Precipitation nulls
    # are safely treated as zero (no rain recorded); temperature and wind
    # nulls are left as NaN, because guessing a temperature would be worse
    # than telling the scorer it doesn't know.
    frame["precip_in"] = frame["precip_in"].fillna(0.0)

    return frame[["time", "temp_f", "precip_in", "wind_mph", "wind_gust_mph"]]


def _get(session, url: str, params: dict) -> dict:
    """One HTTP GET with clear failure messages."""
    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise WeatherUnavailable("Open-Meteo timed out. Check your connection.") from exc
    except requests.exceptions.HTTPError as exc:
        # Open-Meteo puts a useful explanation in the body on a 400, so
        # surface it instead of just the status code.
        detail = ""
        try:
            detail = response.json().get("reason", "")
        except Exception:
            pass
        raise WeatherUnavailable(
            f"Open-Meteo returned {response.status_code}. {detail}".strip()
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise WeatherUnavailable(f"Could not reach Open-Meteo: {exc}") from exc


def fetch_forecast(court: Court, days: int | None = None) -> pd.DataFrame:
    """
    Hourly forecast for one court: today plus the next `days` days.

    `past_days=2` is requested alongside the forecast on purpose. The wet-court
    rule needs to look BACKWARD up to 8 hours, so scoring the very first hour
    of today requires yesterday's rain. Without past_days, an early-morning
    check would wrongly report dry courts after an overnight storm.
    """
    settings = load_settings()
    days = days or settings.get("data", {}).get("forecast_days", 7)

    params = _common_params(court)
    params.update({"forecast_days": days, "past_days": 2})

    payload = _get(forecast_session(), FORECAST_URL, params)
    return _to_frame(payload)


def fetch_history(
    court: Court,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Hourly historical weather for one court between two dates (inclusive).

    Used by history.py to count playable hours per month. This is the slow
    call, three years of hourly data is ~26,000 rows, which is exactly why
    cache.py gives the archive a 30-day lifetime.
    """
    # Clamp the end date: asking past the archive's lag returns empty rows.
    latest_available = dt.date.today() - dt.timedelta(days=ARCHIVE_LAG_DAYS)
    end = min(end, latest_available)
    if start > end:
        raise WeatherUnavailable(
            f"Requested history window starts after the archive's latest "
            f"available date ({latest_available})."
        )

    params = _common_params(court)
    params.update(
        {"start_date": start.isoformat(), "end_date": end.isoformat()}
    )

    payload = _get(archive_session(), ARCHIVE_URL, params)
    return _to_frame(payload)


def fetch_history_years(court: Court, years: int | None = None) -> pd.DataFrame:
    """Convenience wrapper: the last `years` whole years of history."""
    years = years or load_settings().get("data", {}).get("history_years", 3)
    end = dt.date.today() - dt.timedelta(days=ARCHIVE_LAG_DAYS)
    # 365.25 accounts for leap years across a multi-year window.
    start = end - dt.timedelta(days=int(365.25 * years))
    return fetch_history(court, start, end)


def current_conditions(forecast: pd.DataFrame, now: dt.datetime) -> pd.Series | None:
    """
    Pick the row of the forecast matching the current hour.

    Returns None rather than raising if the hour isn't present, a missing row
    is a normal thing that can happen at a day boundary, and the dashboard can
    show 'no data for this hour' more gracefully than it can catch an error.
    """
    target = now.replace(minute=0, second=0, microsecond=0)
    matches = forecast[forecast["time"] == target]
    if matches.empty:
        return None
    return matches.iloc[0]
