"""
scoring.py — the playability model. "Can I physically play here right now?"

THE CORE DESIGN IDEA
Start at 100 points. Each rule looks at the weather and either subtracts
points or vetoes the hour outright. Every deduction is recorded with a
human-readable reason. The final object carries the score together with the
complete list of reasons behind it.

That last part is the whole point. Compare two possible designs:

    BAD:   score = model.predict(weather)   ->  "43"
    THIS:  score = playability(...)         ->  "43 — wet from 0.08in of rain
                                                 2h ago (-31), windy at 17mph
                                                 (-20), chilly at 54F (-6)"

The second one you can argue with. If you go to the court and it's perfectly
dry, you know to raise `drying_rate` in scoring.yaml. A black-box model gives
you nothing to grab onto. For a project you have to explain and defend, an
interpretable model is the stronger choice.

VETO vs PENALTY — an important distinction
  * A PENALTY says "this is worse". It subtracts points and stacks with other
    penalties. Cold + windy is worse than either alone.
  * A VETO says "this is impossible". It sets the score to 0 regardless of
    everything else. It's dark. It's raining. The court is ice.

Getting this distinction right is what stops the model producing silly
answers like "it's 2am and pouring, but it's a comfortable 68F, so: 74/100".
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import pandas as pd

from .config import Court, load_settings
from .daylight import light_situation


@dataclass
class Deduction:
    """One rule's contribution to the score."""

    rule: str  # which rule fired, e.g. "wet_courts"
    points: float  # how many points it removed (positive number)
    reason: str  # plain English, shown in the UI
    is_veto: bool = False  # did this rule force the score to zero?


@dataclass
class Playability:
    """The full result of scoring one court at one hour."""

    score: int  # 0-100, higher is better
    label: str  # "Good" / "Playable" / "Marginal" / "Poor" / "Unplayable"
    color: str
    deductions: list[Deduction] = field(default_factory=list)
    light: str = ""  # "daylight" | "lights" | "dark"
    # The raw weather that produced this, kept so the UI can show it without
    # having to look it up again.
    temp_f: float = math.nan
    wind_mph: float = math.nan
    precip_in: float = math.nan

    @property
    def vetoed(self) -> bool:
        """True if any rule declared this hour physically unplayable."""
        return any(d.is_veto for d in self.deductions)

    @property
    def headline_reason(self) -> str:
        """
        The single most important sentence explaining this score.

        A veto always wins — "it's dark" matters more than "slightly breezy".
        Otherwise the biggest penalty is the most useful thing to say. If
        nothing was deducted at all, the conditions are simply good.
        """
        vetoes = [d for d in self.deductions if d.is_veto]
        if vetoes:
            return vetoes[0].reason
        if not self.deductions:
            return "clear, calm, and comfortable"
        return max(self.deductions, key=lambda d: d.points).reason

    @property
    def full_reason(self) -> str:
        """Every reason, joined — used in tooltips and the detail panel."""
        if not self.deductions:
            return "No deductions: conditions are ideal."
        return "; ".join(
            f"{d.reason} ({'blocked' if d.is_veto else f'-{d.points:.0f}'})"
            for d in self.deductions
        )


# ---------------------------------------------------------------------------
# PRECOMPUTING THE BACKWARD-LOOKING NUMBERS
#
# Two rules need to look BACKWARD in time: wet courts (how much rain recently?)
# and freeze (any precipitation in the last N hours?).
#
# The obvious way to write that is, for each hour, filter the weather frame
# down to the preceding few hours and add them up. That's what this code did
# at first — and scoring 2 years of history took nearly four minutes. The
# reason is that filtering a frame is O(n): doing it once per row makes the
# whole thing O(n²). At 26,000 hours that's ~680 million row comparisons.
#
# The fix is to compute all the backward-looking numbers ONCE for the whole
# frame using vectorized pandas operations, as extra columns. Then scoring an
# hour is just reading values off its own row — no scanning at all. Same
# arithmetic, same results, but seconds instead of minutes.
#
# `prepare()` adds those columns. Both score_hour and score_series call it, so
# there is still only ONE implementation of each rule — the rules just read a
# column instead of recomputing a sum.
# ---------------------------------------------------------------------------

DERIVED_COLUMNS = ("wet_points_raw", "rain_total_in", "rain_hours_ago", "ice_precip_in")


def prepare(history: pd.DataFrame) -> pd.DataFrame:
    """
    Add the precomputed backward-looking columns to an hourly weather frame.

    Assumes rows are hourly and contiguous — which is what Open-Meteo returns.
    We sort defensively because `.shift()` relies on row order: shifting by 1
    means "one row back", and that only equals "one hour back" if the rows are
    in time order with no gaps.

    Columns added:
        wet_points_raw   decayed rain penalty, before the surface multiplier
        rain_total_in    total inches in the lookback window
        rain_hours_ago   hours since the most recent hour with rain
        ice_precip_in    total precipitation in the freeze lookback window
    """
    settings = load_settings()["playability"]
    wet_config = settings["wet_courts"]
    freeze_config = settings["freeze"]

    frame = history.sort_values("time").reset_index(drop=True).copy()
    precip = frame["precip_in"]

    lookback = int(wet_config.get("lookback_hours", 8))
    per_inch = float(wet_config.get("penalty_per_inch", 1200))
    drying_rate = float(wet_config.get("drying_rate", 0.6))

    # The decayed sum:  Σ over k of (rain k hours ago) × per_inch × decay^k
    #
    # Written as a loop over the LOOKBACK (8 iterations), not over the rows
    # (26,000). Each `.shift(k)` operates on the whole column at once, so this
    # is 8 vectorized passes rather than 26,000 individual filters.
    wet_points = pd.Series(0.0, index=frame.index)
    for hours_ago in range(lookback):
        wet_points += precip.shift(hours_ago).fillna(0.0) * per_inch * (
            drying_rate**hours_ago
        )
    frame["wet_points_raw"] = wet_points

    # Total rain in the window, for the human-readable message.
    frame["rain_total_in"] = precip.rolling(lookback, min_periods=1).sum()

    # How many hours since it last rained. Trick: take each row's position,
    # blank it out on dry hours, then forward-fill — every row now carries the
    # position of the most recent wet hour, and the difference is the gap.
    positions = pd.Series(range(len(frame)), index=frame.index, dtype="float64")
    frame["rain_hours_ago"] = positions - positions.where(precip > 0).ffill()

    # Precipitation over the (longer) freeze lookback window.
    ice_lookback = int(freeze_config.get("ice_lookback_hours", 14))
    frame["ice_precip_in"] = precip.rolling(ice_lookback, min_periods=1).sum()

    return frame


def _ensure_prepared(history: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns unless they're already there."""
    if all(column in history.columns for column in DERIVED_COLUMNS):
        return history
    return prepare(history)


def _band_lookup(bands: list[dict], value: float, key_min: str, key_max: str):
    """
    Find the first band whose [min, max) range contains `value`.

    Shared by the temperature and wind rules since both use the same band
    format in scoring.yaml — just with different key names.
    """
    for band in bands:
        if band[key_min] <= value < band[key_max]:
            return band
    return None


def _wet_penalty(settings: dict, court: Court, row: pd.Series) -> Deduction | None:
    """
    RULE 1: how wet is this court from recent rain?

    Reads the decayed rain total that `prepare()` already computed for this
    row. The decay model is explained in detail in scoring.yaml — the short
    version is that rain's penalty shrinks geometrically with every hour that
    passes, because the court is drying.
    """
    config = settings["playability"]["wet_courts"]

    # VETO CHECK: is it raining right now?
    precip_now = float(row["precip_in"])
    if precip_now >= float(config.get("raining_now_veto_inches", 0.01)):
        return Deduction(
            rule="wet_courts",
            points=100,
            reason=f"raining right now ({precip_now:.2f}in this hour)",
            is_veto=True,
        )

    # PENALTY: the precomputed decayed total, scaled for how much water this
    # surface holds compared to a hard court.
    surface_factor = float(config.get("surface_multiplier", {}).get(court.surface, 1.0))
    penalty = float(row["wet_points_raw"]) * surface_factor
    penalty = min(penalty, float(config.get("max_penalty", 100)))

    # Below half a point isn't worth putting on screen.
    if penalty < 0.5:
        return None

    hours_ago_value = row["rain_hours_ago"]
    if pd.isna(hours_ago_value):
        timing = "recently"
    elif hours_ago_value < 1:
        timing = "in the last hour"
    elif hours_ago_value < 2:
        timing = "1 hour ago"
    else:
        timing = f"{int(hours_ago_value)} hours ago"

    total_inches = float(row["rain_total_in"])

    return Deduction(
        rule="wet_courts",
        points=penalty,
        reason=f"courts likely damp — {total_inches:.2f}in of rain, last {timing}",
    )


def _temperature_penalty(settings: dict, temp_f: float) -> Deduction | None:
    """RULE 2: temperature bands."""
    if math.isnan(temp_f):
        return None

    bands = settings["playability"]["temperature_bands"]
    band = _band_lookup(bands, temp_f, "min_f", "max_f")
    if band is None or band["penalty"] <= 0:
        return None

    # A penalty of 100 in the bands table means "this is a veto": below
    # freezing, any water on the court has turned solid.
    is_veto = band["penalty"] >= 100
    return Deduction(
        rule="temperature",
        points=float(band["penalty"]),
        reason=f"{band['label']} at {temp_f:.0f}F",
        is_veto=is_veto,
    )


def _wind_penalty(settings: dict, wind_mph: float, gust_mph: float) -> Deduction | None:
    """
    RULE 3: wind bands.

    DESIGN CHOICE: we score the average of sustained wind and gusts, leaning
    toward gusts, rather than sustained wind alone. A steady 12mph is annoying
    but playable; 12mph sustained with 30mph gusts is genuinely not tennis —
    the gust is what ruins the ball toss. Sustained speed alone would rate
    those two identically, which is wrong. The 60/40 weighting toward gusts is
    a judgement call and a reasonable thing to tune.
    """
    if math.isnan(wind_mph):
        return None

    effective_wind = wind_mph
    if not math.isnan(gust_mph) and gust_mph > wind_mph:
        effective_wind = 0.4 * wind_mph + 0.6 * gust_mph

    bands = settings["playability"]["wind_bands"]
    band = _band_lookup(bands, effective_wind, "min_mph", "max_mph")
    if band is None or band["penalty"] <= 0:
        return None

    detail = f"{wind_mph:.0f}mph"
    if not math.isnan(gust_mph) and gust_mph > wind_mph + 3:
        detail += f", gusting {gust_mph:.0f}"

    return Deduction(
        rule="wind",
        points=float(band["penalty"]),
        reason=f"{band['label']} ({detail})",
    )


def _light_deduction(
    settings: dict, court: Court, when: dt.datetime
) -> tuple[Deduction | None, str]:
    """
    RULE 4: daylight and lights.

    Returns both the deduction and the light situation string, since the
    caller wants to record the situation on the result either way.
    """
    situation, explanation = light_situation(court, when)
    daylight_config = settings["playability"]["daylight"]

    if situation == "dark":
        return (
            Deduction(rule="daylight", points=100, reason=explanation, is_veto=True),
            situation,
        )

    if situation == "lights":
        penalty = float(daylight_config.get("lights_penalty", 6))
        return (
            Deduction(rule="daylight", points=penalty, reason=explanation),
            situation,
        )

    return None, situation  # full daylight: no deduction


def _freeze_veto(settings: dict, row: pd.Series, temp_f: float) -> Deduction | None:
    """
    RULE 5: ice risk — the combination rule.

    Cold alone is a penalty. Wet alone is a penalty. But cold AND recently wet
    means the court may be iced, which makes it a safety issue.
    This is the rule that most justifies the whole rules-based approach: it
    encodes a specific piece of local knowledge (Reno gets clear freezing
    nights after wet days) that no generic weather score would capture.
    """
    if math.isnan(temp_f):
        return None

    config = settings["playability"]["freeze"]
    if temp_f > float(config.get("ice_risk_temp_f", 36)):
        return None

    lookback = int(config.get("ice_lookback_hours", 14))
    recent_precip = float(row["ice_precip_in"])
    if recent_precip < float(config.get("ice_precip_inches", 0.01)):
        return None

    return Deduction(
        rule="freeze",
        points=100,
        reason=(
            f"ice risk — {temp_f:.0f}F with {recent_precip:.2f}in of "
            f"precipitation in the last {lookback}h"
        ),
        is_veto=True,
    )


def _label_for(settings: dict, score: int) -> tuple[str, str]:
    """Turn a numeric score into (label, color) using scoring.yaml."""
    for entry in settings["playability"]["labels"]:
        if score >= entry["min_score"]:
            return entry["label"], entry.get("color", "#8b949e")
    return "Unknown", "#8b949e"


def _score_row(court: Court, row: pd.Series, settings: dict) -> Playability:
    """
    Score one already-prepared row of weather.

    Split out from score_hour so that score_series can loop over rows directly
    without looking each one up by timestamp — that lookup is a full scan of
    the frame, and doing it per row is the other half of the O(n²) problem
    described above `prepare()`.
    """
    when = row["time"]
    temp_f = float(row["temp_f"])
    wind_mph = float(row["wind_mph"])
    gust_mph = float(row["wind_gust_mph"])
    precip_in = float(row["precip_in"])

    # Run every rule. Order here doesn't affect the result — penalties sum and
    # any veto zeroes the score — but it does set the order reasons appear in.
    light_deduction, situation = _light_deduction(settings, court, when)
    candidate_rules = [
        _wet_penalty(settings, court, row),
        _freeze_veto(settings, row, temp_f),
        _temperature_penalty(settings, temp_f),
        _wind_penalty(settings, wind_mph, gust_mph),
        light_deduction,
    ]
    deductions = [d for d in candidate_rules if d is not None]

    # Combine: any veto forces zero, otherwise subtract from the starting score.
    if any(d.is_veto for d in deductions):
        score = 0
    else:
        starting = float(settings["playability"].get("starting_score", 100))
        score = starting - sum(d.points for d in deductions)

    # Clamp into 0-100 and round. Clamping matters because several stacked
    # penalties can easily total more than 100.
    score_int = int(round(max(0.0, min(100.0, score))))
    label, color = _label_for(settings, score_int)

    return Playability(
        score=score_int,
        label=label,
        color=color,
        deductions=deductions,
        light=situation,
        temp_f=temp_f,
        wind_mph=wind_mph,
        precip_in=precip_in,
    )


def score_hour(
    court: Court,
    when: dt.datetime,
    history: pd.DataFrame,
) -> Playability:
    """
    Score one court at one specific hour. This is the main entry point.

    Args:
        court:   which court (its surface and lights matter)
        when:    the hour to score, timezone-aware, minutes zeroed
        history: hourly weather covering `when` AND the hours before it —
                 the backward-looking rules need that runway

    Returns a Playability carrying the score and every reason behind it.
    """
    prepared = _ensure_prepared(history)

    matches = prepared[prepared["time"] == when]
    if matches.empty:
        return Playability(
            score=0,
            label="Unknown",
            color="#8b949e",
            deductions=[
                Deduction("data", 0, "no weather data for this hour", is_veto=True)
            ],
        )

    return _score_row(court, matches.iloc[0], load_settings())


def score_series(court: Court, history: pd.DataFrame) -> pd.DataFrame:
    """
    Score every hour in a weather frame at once.

    Returns a DataFrame with one row per hour: time, score, label, reason,
    and the raw weather. This is what the 7-day heatmap and the historical
    view are both built from.

    DESIGN CHOICE: the first few hours of the frame get lower-confidence
    scores, because the wet-courts rule can only look back as far as the data
    goes. We don't drop them — the caller decides what to show — but this is
    why weather.py always requests `past_days=2` alongside the forecast.
    """
    prepared = _ensure_prepared(history)
    settings = load_settings()

    rows = []
    # itertuples would be faster still, but named-column access keeps the rule
    # functions readable, and this is already fast enough at this data size.
    for _, weather_row in prepared.iterrows():
        result = _score_row(court, weather_row, settings)
        rows.append(
            {
                "time": weather_row["time"],
                "score": result.score,
                "label": result.label,
                "color": result.color,
                "reason": result.headline_reason,
                "full_reason": result.full_reason,
                "light": result.light,
                "vetoed": result.vetoed,
                "temp_f": result.temp_f,
                "wind_mph": result.wind_mph,
                "precip_in": result.precip_in,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        # Return an empty frame with the right columns so callers can rely on
        # the shape rather than special-casing emptiness everywhere.
        return pd.DataFrame(
            columns=[
                "time", "score", "label", "color", "reason",
                "full_reason", "light", "vetoed", "temp_f", "wind_mph", "precip_in",
            ]
        )
    return frame


def playable_threshold() -> int:
    """The score at or above which an hour counts as 'playable'."""
    return int(load_settings()["playability"].get("playable_threshold", 50))
