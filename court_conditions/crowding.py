"""
crowding.py — the crowding model. "Will a court actually be free?"

This is a completely separate model from scoring.py, and that separation is
deliberate. See the long comment in scoring.yaml under `crowding:` for the
reasoning, but the short version: playability and crowding often point in
OPPOSITE directions. The nicest hour of the week is also the most contested
one. Collapsing both into a single "should I go" number would throw away the
tension that makes the question interesting.

BE HONEST ABOUT WHAT THIS IS
The playability model is grounded in measured data — real temperature, real
rainfall. The crowding model is grounded in your guesses, written down in
courts.yaml. It is a structured way of encoding what you already know as a
coach. It has no way to know that a tournament took over the Tennis Center
this Saturday.

That's a genuine limitation, and the dashboard states it openly. The honest
framing is: playability is a forecast, crowding is a reminder.

OUTPUT: 0-100 where HIGHER MEANS MORE CROWDED.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .config import WEEKDAY_NAMES, Court, load_settings


@dataclass
class Crowding:
    """The result of estimating crowding for one court at one hour."""

    score: int  # 0-100, higher = more crowded
    label: str  # "Packed" / "Busy" / "Some players" / "Likely empty"
    expected_free_courts: float  # rough count of courts likely open
    reason: str  # why this number

    @property
    def likely_available(self) -> bool:
        """Is at least one court probably open?"""
        return self.expected_free_courts >= 1.0


def _base_level(court: Court, when: dt.datetime) -> tuple[float, str]:
    """
    Step 1: start from the busy-hours blocks you wrote in courts.yaml.

    If any block covers this weekday and hour, use its `level`. If several
    overlap, the busiest one wins — overlapping blocks usually mean two
    different things happen at once (a league AND lessons), which makes the
    place busier than either one alone would.
    """
    settings = load_settings()["crowding"]
    weekday_index = when.weekday()  # Monday = 0, matching WEEKDAY_NAMES

    matching = [
        block
        for block in court.busy_hours
        if block.applies_to(weekday_index, when.hour)
    ]

    if not matching:
        base = float(settings.get("base_level", 0.12))
        return base, "outside your noted busy hours"

    busiest = max(matching, key=lambda b: b.level)
    explanation = busiest.why or "a busy block you configured"
    return busiest.level, explanation


def estimate(
    court: Court,
    when: dt.datetime,
    playability_score: int | None = None,
) -> Crowding:
    """
    Estimate crowding for one court at one hour.

    Args:
        court:              which court (its busy_hours and num_courts matter)
        when:               the hour to estimate
        playability_score:  the 0-100 playability for the same hour. Optional,
                            but passing it makes the estimate much better —
                            nobody is at the courts during a hailstorm no
                            matter what the schedule says.

    The calculation is four multiplications on a 0-1 occupancy level:
        base (from your busy_hours) x day-of-week x month x weather
    then scaled to 0-100.
    """
    settings = load_settings()["crowding"]

    # Step 1: your configured baseline for this weekday and hour.
    level, base_reason = _base_level(court, when)

    # Step 2: day-of-week multiplier. Saturday draws more people than Friday
    # even at the identical hour.
    weekday_name = WEEKDAY_NAMES[when.weekday()]
    day_factor = float(settings.get("day_multiplier", {}).get(weekday_name, 1.0))
    level *= day_factor

    # Step 3: seasonal multiplier. This is the big one in Reno — a 5pm
    # Wednesday in July and a 5pm Wednesday in January are wildly different
    # questions, and the busy_hours config alone can't express that because
    # it has no notion of the calendar.
    month_factor = float(settings.get("month_multiplier", {}).get(when.month, 1.0))
    level *= month_factor

    # Step 4: weather response. Nice weather pulls people outdoors.
    #
    # This is a LINEAR INTERPOLATION: at playability 0 we use the
    # `at_terrible_weather` multiplier, at 100 we use `at_perfect_weather`,
    # and in between we slide proportionally. The formula
    #     low + (high - low) * (score / 100)
    # is worth recognising — it's the standard way to map one range onto
    # another and shows up constantly once you start looking for it.
    weather_reason = ""
    weather_config = settings.get("weather_response", {})
    if weather_config.get("enabled", True) and playability_score is not None:
        low = float(weather_config.get("at_terrible_weather", 0.15))
        high = float(weather_config.get("at_perfect_weather", 1.15))
        weather_factor = low + (high - low) * (playability_score / 100.0)
        level *= weather_factor

        if playability_score < 35:
            weather_reason = "; bad conditions will keep most people home"
        elif playability_score > 80:
            weather_reason = "; great conditions will draw a crowd"

    # Clamp to 0-1. The multipliers can stack above 1.0 (a busy block at 0.95
    # on a Saturday in July), but "more than every court taken" is meaningless.
    level = max(0.0, min(1.0, level))

    score = int(round(level * 100))

    # Convert occupancy into an actual court count, which is the number you
    # care about standing in the parking lot. 8 courts at 75% busy means
    # roughly 2 open.
    expected_free = court.num_courts * (1.0 - level)

    label = _label_for(score)

    # Assemble the explanation from the pieces that actually moved the number.
    parts = [base_reason]
    if month_factor < 0.7:
        parts.append("quiet season")
    elif month_factor > 1.15:
        parts.append("peak season")
    if day_factor > 1.1:
        parts.append("weekend")
    reason = ", ".join(parts) + weather_reason

    return Crowding(
        score=score,
        label=label,
        expected_free_courts=round(expected_free, 1),
        reason=reason,
    )


def _label_for(score: int) -> str:
    """Turn a crowding number into words, using the table in scoring.yaml."""
    for entry in load_settings()["crowding"]["labels"]:
        if score >= entry["min_score"]:
            return entry["label"]
    return "Unknown"


def estimate_series(court: Court, scored: "object") -> list[Crowding]:
    """
    Estimate crowding for every hour in a scored playability frame.

    Takes the DataFrame that scoring.score_series produced so the weather
    response can use each hour's own playability score.
    """
    results = []
    for _, row in scored.iterrows():
        results.append(estimate(court, row["time"], int(row["score"])))
    return results
