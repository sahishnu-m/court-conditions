"""
recommend.py: turns scores into answers. "When should I actually go?"

scoring.py tells you how good every hour is. This file answers the questions
you'd actually ask standing at the courts with your phone out:

    Can I play right now?
    When's the best time today?
    What's the best slot this week?

DESIGN CHOICE: combining playability and crowding, but only HERE.
The two models stay strictly separate in scoring.py and crowding.py. This is
the one place they're weighed against each other, and even here the combined
number is used purely for RANKING options, while the display always shows the
two real numbers. That way the worst a bad combination can do is reorder a
list, leaving the conditions you actually read still accurate.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from . import crowding as crowding_module
from .config import Court
from .daylight import format_clock
from .scoring import Playability, score_hour


@dataclass
class Slot:
    """One candidate hour to play, with both models' verdicts."""

    court: Court
    when: dt.datetime
    playability: Playability
    crowding: crowding_module.Crowding

    @property
    def combined_rank(self) -> float:
        """
        The ranking number used to sort options. NEVER displayed.

        Weighted 70/30 toward playability. The reasoning: bad weather makes
        tennis impossible, while a crowd makes it merely inconvenient, you
        can wait twenty minutes for a court, but you can't wait out a
        thunderstorm. So playability dominates, with crowding as a tiebreaker
        that separates two otherwise-equally-nice hours.

        A vetoed hour ranks below everything, always.
        """
        if self.playability.score == 0:
            return -1.0
        freeness = 100 - self.crowding.score
        return 0.7 * self.playability.score + 0.3 * freeness

    @property
    def time_label(self) -> str:
        """e.g. 'Thu 5:00pm'."""
        return f"{self.when.strftime('%a')} {format_clock(self.when)}"

    @property
    def summary(self) -> str:
        """One line combining both verdicts, for a list item."""
        return (
            f"{self.playability.score}/100 {self.playability.label} · "
            f"{self.crowding.label}, about "
            f"{self.crowding.expected_free_courts:.0f} of "
            f"{self.court.num_courts} courts free"
        )


def build_slot(court: Court, when: dt.datetime, weather: pd.DataFrame) -> Slot:
    """Score one hour at one court with both models."""
    playability = score_hour(court, when, weather)
    crowd = crowding_module.estimate(court, when, playability.score)
    return Slot(court=court, when=when, playability=playability, crowding=crowd)


def right_now(court: Court, weather: pd.DataFrame, now: dt.datetime) -> Slot:
    """
    The headline answer: can I play at this court right now?

    `now` is rounded down to the hour because the weather data is hourly.
    2:47pm is scored using the 2:00pm row, the closest thing we have.
    """
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    return build_slot(court, current_hour, weather)


def best_today(
    court: Court,
    weather: pd.DataFrame,
    now: dt.datetime,
    min_score: int = 40,
) -> list[Slot]:
    """
    The best remaining hours today, best first.

    Only looks FORWARD from the current hour, telling you that 9am was great
    when it's 4pm is not useful. Returns an empty list when nothing today
    clears `min_score`, which is itself a useful answer ("don't bother today").
    """
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    end_of_day = current_hour.replace(hour=23, minute=59)

    candidates = weather[
        (weather["time"] >= current_hour) & (weather["time"] <= end_of_day)
    ]

    slots = [build_slot(court, t, weather) for t in candidates["time"]]
    playable = [s for s in slots if s.playability.score >= min_score]
    return sorted(playable, key=lambda s: s.combined_rank, reverse=True)


def best_this_week(
    court: Court,
    weather: pd.DataFrame,
    now: dt.datetime,
    min_score: int = 40,
    per_day: int = 1,
) -> list[Slot]:
    """
    The best slot on each of the next several days.

    DESIGN CHOICE: one result per day rather than a flat "top 10 hours".

    A flat top-10 would almost always return ten consecutive hours from
    whichever day happens to be nicest, 10am, 11am, 12pm, 1pm... on Saturday.
    That's technically correct and practically useless; you can only play once.
    Grouping by day gives you a genuine menu of options to plan around.
    """
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    future = weather[weather["time"] >= current_hour]

    best_per_day: list[Slot] = []
    for _, day_rows in future.groupby(future["time"].dt.date):
        slots = [build_slot(court, t, weather) for t in day_rows["time"]]
        playable = [s for s in slots if s.playability.score >= min_score]
        if not playable:
            continue
        playable.sort(key=lambda s: s.combined_rank, reverse=True)
        best_per_day.extend(playable[:per_day])

    return sorted(best_per_day, key=lambda s: s.when)


def best_court_right_now(
    slots_by_court: dict[str, Slot],
) -> Slot | None:
    """
    Given one 'right now' slot per court, pick the best place to go.

    Used for the "if not here, then where?" line on the dashboard. Returns
    None if every court is vetoed, because when conditions are that bad the
    honest answer is "nowhere, stay home". Surfacing a least-bad option there
    would dress up a wasted trip as a recommendation.
    """
    playable = [s for s in slots_by_court.values() if s.playability.score > 0]
    if not playable:
        return None
    return max(playable, key=lambda s: s.combined_rank)
