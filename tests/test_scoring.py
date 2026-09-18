"""
Tests for the scoring rules.

WHY TESTS MATTER HERE SPECIFICALLY
The whole selling point of a rules-based model is that you can tune it. But
tuning is dangerous without tests: you nudge `drying_rate` to fix one case and
silently break the freeze rule. These tests pin down the BEHAVIOUR you care
about ("freezing weather stays unplayable") while leaving the exact numbers
free, so you can keep tuning the weights and still hear about it immediately
when a change breaks something fundamental.

Every test builds its own fake weather instead of calling the API. Tests that
need the network are slow, flaky, and can't test a scenario like "0.3 inches
of rain four hours ago" on demand.

Run them with:   python -m pytest tests/ -v
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from court_conditions.config import Court, get_timezone, load_courts  # noqa: E402
from court_conditions.crowding import estimate  # noqa: E402
from court_conditions.scoring import score_hour  # noqa: E402

TZ = ZoneInfo(get_timezone())


def make_court(**overrides) -> Court:
    """A plain test court. Override any field per test."""
    defaults = dict(
        id="test",
        name="Test Court",
        city="Reno",
        lat=39.5296,
        lon=-119.8138,
        num_courts=4,
        surface="hard",
        lights=True,
        lights_out=22,
        verified=True,
        notes="",
        busy_hours=(),
    )
    defaults.update(overrides)
    return Court(**defaults)


def make_weather(
    when: dt.datetime,
    temp_f: float = 70.0,
    wind_mph: float = 3.0,
    gust_mph: float = 5.0,
    precip_by_hours_ago: dict[int, float] | None = None,
    hours_of_history: int = 16,
) -> pd.DataFrame:
    """
    Build a fake hourly weather frame ending at `when`.

    `precip_by_hours_ago` lets a test say "0.1 inches fell 3 hours ago" as
    {3: 0.1}, which reads much closer to the scenario being described than
    hand-building a DataFrame would.
    """
    precip_by_hours_ago = precip_by_hours_ago or {}
    rows = []
    for hours_ago in range(hours_of_history, -1, -1):
        rows.append(
            {
                "time": when - dt.timedelta(hours=hours_ago),
                "temp_f": temp_f,
                "precip_in": precip_by_hours_ago.get(hours_ago, 0.0),
                "wind_mph": wind_mph,
                "wind_gust_mph": gust_mph,
            }
        )
    return pd.DataFrame(rows)


# A summer midday hour, which is safely in full daylight in Reno so daylight
# never interferes with tests that are about something else.
GOOD_HOUR = dt.datetime(2026, 7, 15, 13, 0, tzinfo=TZ)


class TestPerfectConditions:
    def test_ideal_weather_scores_high(self):
        court = make_court()
        weather = make_weather(GOOD_HOUR, temp_f=72, wind_mph=3, gust_mph=5)
        result = score_hour(court, GOOD_HOUR, weather)

        assert result.score >= 90
        assert not result.vetoed
        assert result.light == "daylight"

    def test_perfect_conditions_have_no_deductions(self):
        court = make_court()
        weather = make_weather(GOOD_HOUR, temp_f=70, wind_mph=2, gust_mph=3)
        result = score_hour(court, GOOD_HOUR, weather)
        assert result.deductions == []
        assert "ideal" in result.full_reason.lower()


class TestRain:
    def test_raining_now_is_a_veto(self):
        court = make_court()
        weather = make_weather(GOOD_HOUR, precip_by_hours_ago={0: 0.05})
        result = score_hour(court, GOOD_HOUR, weather)

        assert result.score == 0
        assert result.vetoed
        assert "raining right now" in result.headline_reason

    def test_recent_rain_penalises_but_does_not_veto(self):
        court = make_court()
        weather = make_weather(GOOD_HOUR, precip_by_hours_ago={2: 0.10})
        result = score_hour(court, GOOD_HOUR, weather)

        assert 0 < result.score < 90, "damp courts should cost points"
        assert not result.vetoed, "rain 2h ago should deduct points and stop there"

    def test_older_rain_hurts_less_than_recent_rain(self):
        """The drying curve is the core of the wet-court rule — verify it decays."""
        court = make_court()
        recent = score_hour(
            court, GOOD_HOUR, make_weather(GOOD_HOUR, precip_by_hours_ago={1: 0.1})
        )
        older = score_hour(
            court, GOOD_HOUR, make_weather(GOOD_HOUR, precip_by_hours_ago={6: 0.1})
        )
        assert older.score > recent.score

    def test_rain_beyond_lookback_is_forgotten(self):
        court = make_court()
        weather = make_weather(
            GOOD_HOUR, precip_by_hours_ago={15: 0.5}, hours_of_history=20
        )
        result = score_hour(court, GOOD_HOUR, weather)
        assert result.score >= 90, "rain 15h ago is outside the 8h lookback"

    def test_clay_is_penalised_more_than_hard_court(self):
        hard = score_hour(
            make_court(surface="hard"),
            GOOD_HOUR,
            make_weather(GOOD_HOUR, precip_by_hours_ago={3: 0.1}),
        )
        clay = score_hour(
            make_court(surface="clay"),
            GOOD_HOUR,
            make_weather(GOOD_HOUR, precip_by_hours_ago={3: 0.1}),
        )
        assert clay.score < hard.score


class TestTemperature:
    def test_freezing_is_a_veto(self):
        court = make_court()
        weather = make_weather(GOOD_HOUR, temp_f=25)
        result = score_hour(court, GOOD_HOUR, weather)
        assert result.score == 0
        assert result.vetoed

    def test_extreme_heat_hurts_but_is_not_a_veto(self):
        court = make_court()
        weather = make_weather(GOOD_HOUR, temp_f=104)
        result = score_hour(court, GOOD_HOUR, weather)
        assert result.score < 60
        assert not result.vetoed, "extreme heat is miserable but still playable"

    @pytest.mark.parametrize("temp_f", [60, 70, 80])
    def test_ideal_band_costs_nothing(self, temp_f):
        court = make_court()
        weather = make_weather(GOOD_HOUR, temp_f=temp_f)
        result = score_hour(court, GOOD_HOUR, weather)
        assert not any(d.rule == "temperature" for d in result.deductions)


class TestWind:
    def test_strong_wind_tanks_the_score(self):
        court = make_court()
        weather = make_weather(GOOD_HOUR, wind_mph=32, gust_mph=40)
        result = score_hour(court, GOOD_HOUR, weather)
        assert result.score < 40

    def test_gusts_matter_more_than_average_wind(self):
        """
        The design decision that gusts dominate the wind rule: two hours with
        identical sustained wind should diverge once one of them is gusting.
        """
        court = make_court()
        steady = score_hour(
            court, GOOD_HOUR, make_weather(GOOD_HOUR, wind_mph=12, gust_mph=13)
        )
        gusty = score_hour(
            court, GOOD_HOUR, make_weather(GOOD_HOUR, wind_mph=12, gust_mph=34)
        )
        assert gusty.score < steady.score


class TestDaylightAndLights:
    def test_dark_with_no_lights_is_a_veto(self):
        court = make_court(lights=False, lights_out=None)
        midnight = dt.datetime(2026, 7, 15, 23, 0, tzinfo=TZ)
        result = score_hour(court, midnight, make_weather(midnight))
        assert result.score == 0
        assert "no lights" in result.headline_reason

    def test_lights_make_an_evening_playable(self):
        court = make_court(lights=True, lights_out=22)
        evening = dt.datetime(2026, 7, 15, 21, 0, tzinfo=TZ)
        result = score_hour(court, evening, make_weather(evening))
        assert result.score > 0
        assert result.light == "lights"

    def test_past_lights_out_is_a_veto(self):
        court = make_court(lights=True, lights_out=22)
        late = dt.datetime(2026, 7, 15, 23, 0, tzinfo=TZ)
        result = score_hour(court, late, make_weather(late))
        assert result.score == 0

    def test_winter_afternoon_is_dark_but_summer_is_not(self):
        """
        Reno's ~4-hour seasonal daylight swing is the biggest single factor in
        this model. 6pm in July is daylight; 6pm in December is not.
        """
        court = make_court(lights=False, lights_out=None)
        summer = dt.datetime(2026, 7, 15, 18, 0, tzinfo=TZ)
        winter = dt.datetime(2026, 12, 15, 18, 0, tzinfo=TZ)

        assert score_hour(court, summer, make_weather(summer)).light == "daylight"
        assert score_hour(court, winter, make_weather(winter)).light == "dark"


class TestFreezeRule:
    def test_cold_plus_recent_wet_is_an_ice_veto(self):
        """
        The combination rule. 35F alone earns a penalty, and rain alone earns
        a penalty. Together they mean ice, which is a veto.
        """
        court = make_court()
        # 10am in March, safely after sunrise, so daylight isn't the cause.
        when = dt.datetime(2026, 3, 15, 10, 0, tzinfo=TZ)
        weather = make_weather(
            when, temp_f=34, precip_by_hours_ago={8: 0.05}, hours_of_history=16
        )
        result = score_hour(court, when, weather)

        assert result.score == 0
        assert "ice risk" in result.headline_reason

    def test_cold_and_dry_is_only_a_penalty(self):
        court = make_court()
        when = dt.datetime(2026, 3, 15, 10, 0, tzinfo=TZ)
        result = score_hour(court, when, make_weather(when, temp_f=34))
        assert result.score == 0 or not any(
            d.rule == "freeze" for d in result.deductions
        ), "dry cold should not trigger the ice rule"


class TestExplanations:
    def test_every_deduction_carries_a_reason(self):
        court = make_court()
        weather = make_weather(
            GOOD_HOUR, temp_f=45, wind_mph=18, gust_mph=25,
            precip_by_hours_ago={3: 0.05},
        )
        result = score_hour(court, GOOD_HOUR, weather)

        assert result.deductions, "these conditions should deduct something"
        for deduction in result.deductions:
            assert deduction.reason.strip(), f"{deduction.rule} has no explanation"
            assert deduction.rule.strip()

    def test_score_is_always_in_range(self):
        """Several stacked penalties can exceed 100 — the result must clamp."""
        court = make_court()
        weather = make_weather(
            GOOD_HOUR, temp_f=38, wind_mph=27, gust_mph=35,
            precip_by_hours_ago={1: 0.3},
        )
        result = score_hour(court, GOOD_HOUR, weather)
        assert 0 <= result.score <= 100


class TestCrowding:
    def test_busy_hours_config_raises_crowding(self):
        from court_conditions.config import BusyBlock

        quiet_court = make_court()
        busy_court = make_court(
            busy_hours=(
                BusyBlock(
                    days=("wed",), start_hour=17, end_hour=20, level=0.95, why="league"
                ),
            )
        )
        # 2026-07-15 is a Wednesday.
        when = dt.datetime(2026, 7, 15, 18, 0, tzinfo=TZ)

        assert estimate(busy_court, when, 90).score > estimate(quiet_court, when, 90).score

    def test_bad_weather_empties_the_courts(self):
        """The one direction the two models are linked: playability -> crowding."""
        court = make_court()
        when = dt.datetime(2026, 7, 15, 18, 0, tzinfo=TZ)
        assert estimate(court, when, 10).score < estimate(court, when, 95).score

    def test_winter_is_quieter_than_summer(self):
        court = make_court()
        july = dt.datetime(2026, 7, 15, 17, 0, tzinfo=TZ)
        january = dt.datetime(2026, 1, 14, 17, 0, tzinfo=TZ)
        assert estimate(court, january, 70).score < estimate(court, july, 70).score

    def test_free_court_count_never_exceeds_the_courts_available(self):
        court = make_court(num_courts=4)
        when = dt.datetime(2026, 1, 14, 3, 0, tzinfo=TZ)
        result = estimate(court, when, 0)
        assert 0 <= result.expected_free_courts <= 4


class TestConfigLoads:
    def test_real_courts_yaml_parses(self):
        """Catches a YAML typo before it reaches the dashboard."""
        courts = load_courts()
        assert len(courts) >= 6, "the brief asked for 6-10 seeded courts"
        for court in courts:
            assert -90 <= court.lat <= 90
            assert -180 <= court.lon <= 180
            assert court.num_courts > 0
            assert court.surface in {"hard", "clay", "other"}
            # Lights without a shutoff hour would let the app think you can
            # play at 4am under floodlights.
            if court.lights:
                assert court.lights_out is not None

    def test_courts_are_actually_near_reno(self):
        """A sign-flipped longitude is an easy typo and puts a court in China."""
        for court in load_courts():
            assert 39.2 <= court.lat <= 39.8, f"{court.id} latitude is not near Reno"
            assert -120.3 <= court.lon <= -119.5, f"{court.id} longitude is not near Reno"
