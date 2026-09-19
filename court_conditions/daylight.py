"""
daylight.py: when is the sun up at a given court?

DESIGN CHOICE: this is computed locally, on your own machine.

Sunrise and sunset are pure astronomy. Given a latitude, a longitude, and a
date, the answer is a fixed calculation that has been known for centuries,
there is nothing to look up. Calling an API for it would add a network round
trip, a failure mode, and a rate limit in exchange for nothing. The `astral`
library does the math in microseconds, offline.

That also means the app degrades gracefully: if Open-Meteo is unreachable but
the forecast is cached, daylight still works perfectly, because it never
needed the network in the first place.

WHY DAYLIGHT MATTERS SO MUCH HERE
Reno is at about 39.5 degrees north. In June the sun sets around 8:30pm; in
December it's dark by 4:45pm. That is a nearly four-hour swing in when you
can play, and it's the single largest seasonal factor in this whole model,
bigger than temperature. A court without lights simply loses half its useful
hours in winter, and the app has to know that.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun

from .config import Court, get_timezone, load_settings


def format_clock(when: dt.datetime) -> str:
    """
    Format a time as "6:42pm".

    Written by hand instead of using strftime("%-I:%M%p") because the `%-I`
    flag (hour without a leading zero) only exists on Linux and macOS, it
    raises an error on Windows. Since you develop on Windows and Streamlit
    Cloud runs Linux, anything platform-specific would work on one and break
    on the other. Doing it manually works everywhere.
    """
    hour_24 = when.hour
    hour_12 = hour_24 % 12 or 12  # 0 and 12 both display as 12
    suffix = "am" if hour_24 < 12 else "pm"
    return f"{hour_12}:{when.minute:02d}{suffix}"


@lru_cache(maxsize=4096)
def _sun_times_cached(
    lat: float, lon: float, date_ordinal: int, tz_name: str
) -> tuple[dt.datetime, dt.datetime]:
    """
    Sunrise and sunset for one place on one day.

    Cached because scoring a 7-day heatmap across 9 courts asks for the same
    (court, date) pair 24 times over, once per hour of that day. The
    calculation is cheap but not free, and 24x is 24x.

    The cache key uses `date_ordinal` (a plain int) rather than a date object
    purely because lru_cache keys must be hashable and simple types make the
    caching behaviour obvious.
    """
    date = dt.date.fromordinal(date_ordinal)
    timezone = ZoneInfo(tz_name)

    # LocationInfo wants a name and region for display purposes only; the math
    # uses just the coordinates and timezone.
    location = LocationInfo(
        name="court", region="NV", timezone=tz_name, latitude=lat, longitude=lon
    )

    times = sun(location.observer, date=date, tzinfo=timezone)
    return times["sunrise"], times["sunset"]


def sun_times(court: Court, date: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Sunrise and sunset at this court on this date, as local datetimes."""
    return _sun_times_cached(court.lat, court.lon, date.toordinal(), get_timezone())


def playable_light_window(
    court: Court, date: dt.date
) -> tuple[dt.datetime, dt.datetime]:
    """
    The window during which there is enough NATURAL light to play.

    This is deliberately wider than sunrise-to-sunset. You can see a tennis
    ball for a while before the sun clears the horizon and for a while after
    it drops below, that's civil twilight. The exact minutes are tunable in
    scoring.yaml (`minutes_before_sunrise` / `minutes_after_sunset`) because
    how long you can genuinely see the ball depends on your eyes and on how
    hemmed-in by hills a particular court is. Reno courts on the west side sit
    under the Sierra and lose usable light noticeably earlier than the
    almanac's sunset time, that's a real thing you could tune for.
    """
    daylight_config = load_settings()["playability"]["daylight"]
    sunrise, sunset = sun_times(court, date)

    return (
        sunrise - dt.timedelta(minutes=daylight_config.get("minutes_before_sunrise", 15)),
        sunset + dt.timedelta(minutes=daylight_config.get("minutes_after_sunset", 25)),
    )


def is_daylight(court: Court, when: dt.datetime) -> bool:
    """True if there is enough natural light at this court at this moment."""
    start, end = playable_light_window(court, when.date())
    return start <= when <= end


def light_situation(court: Court, when: dt.datetime) -> tuple[str, str]:
    """
    Classify the lighting at a given hour.

    Returns (situation, explanation) where situation is one of:
        "daylight", natural light, ideal
        "lights", dark, but this court has working lights and they're on
        "dark", dark with no usable lights; you cannot play

    Returning the reason alongside the verdict is a pattern used throughout
    this project: every judgement carries its own explanation, so the
    dashboard never has to reconstruct why something was decided.
    """
    if is_daylight(court, when):
        _, sunset = sun_times(court, when.date())
        return "daylight", f"daylight (sunset {format_clock(sunset)})"

    if not court.lights:
        return "dark", "dark and this court has no lights"

    # The court has lights, but are they still on at this hour?
    if court.lights_out is not None and when.hour >= court.lights_out:
        return "dark", f"lights shut off at {court.lights_out}:00"

    # Very early morning: dark, and lights generally stay off before dawn.
    # Treating 4am as unlit is a judgement call about how parks behave, # they
    # rarely run court lights overnight. Change this if yours do.
    if when.hour < 5:
        return "dark", "before dawn; court lights are not on overnight"

    return "lights", "after dark, playing under lights"
