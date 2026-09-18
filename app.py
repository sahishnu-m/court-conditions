"""
The web app.

Run it locally:
    streamlit run app.py

Streamlit reruns this whole script from top to bottom every time you click
something. The @st.cache_data decorators make sure the slow parts (fetching
weather and scoring thousands of hours) only happen once.

Look and feel comes from .streamlit/config.toml, so there is no CSS injected
here beyond the two small st.html blocks that draw the score bar and the
key/value rows.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from court_conditions import recommend
from court_conditions.cache import cache_status, clear_cache
from court_conditions.config import find_court, get_timezone, load_courts, load_settings
from court_conditions.daylight import format_clock, sun_times
from court_conditions.history import (
    monthly_averages,
    monthly_playable_hours,
    seasonal_summary,
)
from court_conditions.scoring import playable_threshold, score_series
from court_conditions.scrape import load_report
from court_conditions.weather import WeatherUnavailable, fetch_forecast

st.set_page_config(page_title="Court Conditions", layout="centered")

MUTED = "#52514e"
INK = "#0b0b0b"

# The five score colours, matching the labels in config/scoring.yaml. Charts
# read this list so the heatmap and the headline number always agree.
SCORE_DOMAIN = [0, 40, 60, 80, 100]
SCORE_RANGE = ["#c0392b", "#eb6834", "#e0a030", "#6aa84f", "#1baf7a"]
SCORE_SCALE = alt.Scale(domain=SCORE_DOMAIN, range=SCORE_RANGE, type="linear")

TIMEZONE = ZoneInfo(get_timezone())


# ---------------------------------------------------------------------------
# Loading (cached)
#
# These take a court id string rather than a Court object, because Streamlit
# hashes the arguments to build the cache key and plain strings hash
# predictably. This is a second cache on top of requests_cache in cache.py:
# that one skips the network call, this one skips re-scoring on every click.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(court_id: str) -> pd.DataFrame:
    return fetch_forecast(find_court(court_id))


@st.cache_data(ttl=1800, show_spinner=False)
def load_scored(court_id: str) -> pd.DataFrame:
    return score_series(find_court(court_id), load_forecast(court_id))


@st.cache_data(ttl=86400, show_spinner=False)
def load_history(court_id: str, years: int) -> pd.DataFrame:
    return monthly_playable_hours(find_court(court_id), years=years)


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

def score_bar(score: int, label: str, colour: str, reason: str) -> None:
    """The headline number, with a bar showing where it sits out of 100."""
    st.html(f"""
    <div style="font-family:inherit;margin:0.5rem 0 0.25rem">
      <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-end">
        <div style="min-width:0">
          <div style="color:{MUTED};font-size:0.9rem">Playability</div>
          <div style="color:{INK};font-size:2.4rem;font-weight:700;line-height:1.1">{score}
            <span style="font-size:1.1rem;font-weight:400;color:{MUTED}">/ 100</span></div>
        </div>
        <div style="min-width:0;text-align:right">
          <div style="color:{MUTED};font-size:0.9rem">Verdict</div>
          <div style="color:{colour};font-size:2.4rem;font-weight:700;line-height:1.1">{label}</div>
        </div>
      </div>
      <div role="img" aria-label="Playability {score} out of 100"
           style="display:flex;height:18px;margin-top:0.5rem;background:#eceae4;border-radius:4px">
        <div style="width:{max(score, 2)}%;background:{colour};border-radius:4px"></div>
      </div>
      <div style="color:{MUTED};font-size:0.9rem;margin-top:0.45rem">{reason}</div>
    </div>""")


def kv_rows(pairs: list[tuple[str, str]]) -> None:
    """A simple two-column list, used for score breakdowns and sun times."""
    body = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:1rem;'
        f'padding:0.4rem 0;border-bottom:1px solid #e6e5e0">'
        f'<span style="color:{INK}">{k}</span>'
        f'<span style="color:{MUTED};text-align:right">{v}</span></div>'
        for k, v in pairs
    )
    st.html(f'<div style="font-family:inherit;font-size:0.9rem">{body}</div>')


def heatmap(frame: pd.DataFrame, x: str, y: str, tooltip: list, height: int,
            x_sort=None, y_sort=None) -> alt.Chart:
    """
    Shared rect heatmap, so both outlook charts look identical.

    `x_sort` and `y_sort` matter more than they look. Altair orders a
    categorical axis by first appearance when sorting is switched off, and the
    forecast frame starts at the CURRENT hour. Left alone, that puts 10pm at
    the top of the axis and 6am below it, because 10pm is the first hour in the
    data. Passing an explicit order keeps the axis reading 6am down to 10pm
    whatever time of day the page is loaded.
    """
    return (
        alt.Chart(frame)
        .mark_rect(cornerRadius=2, stroke="#fcfcfb", strokeWidth=1.5)
        .encode(
            x=alt.X(x, sort=x_sort, title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=11, ticks=False, domain=False)),
            y=alt.Y(y, sort=y_sort, title=None,
                    axis=alt.Axis(labelFontSize=11, ticks=False, domain=False)),
            color=alt.Color(
                "score:Q", scale=SCORE_SCALE,
                legend=alt.Legend(title="Score", orient="right", gradientLength=180,
                                  values=[0, 25, 50, 75, 100]),
            ),
            tooltip=tooltip,
        )
        .properties(height=height)
    )


def hour_label(hour: int) -> str:
    """7 -> '7am', 13 -> '1pm'. Written out rather than using strftime's %-I,
    which is a Linux-only flag that raises on Windows."""
    return f"{hour % 12 or 12}{'am' if hour < 12 else 'pm'}"


# Fixed 6am-to-10pm ordering for the hourly axis.
HOUR_ORDER = [hour_label(h) for h in range(6, 23)]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

courts = load_courts()
settings = load_settings()
threshold = playable_threshold()
now = dt.datetime.now(TIMEZONE)

st.title("Court Conditions")
st.markdown(
    f"""
This is a sports analytics project that estimates whether the **{len(courts)} public outdoor
tennis courts** around Reno and Sparks are **playable** and **free** at a given hour.

It combines **hourly weather**, **sunrise and sunset computed for each court's exact
position**, and **typical busy hours**, into a score out of 100 that shows its working for
every point it takes off. Conditions are read live, and the seasonal view scores several
years of past weather with the same rules.

Weather: [Open-Meteo](https://open-meteo.com), free and without an API key, licensed
[CC BY 4.0](https://open-meteo.com/en/license). Times are Reno local, currently
**{now:%A %d %B}, {format_clock(now)}**.
"""
)

# ---------------------------------------------------------------------------
# Part I: right now
# ---------------------------------------------------------------------------

st.header("Part I: Can I play right now?", divider="gray")

col1, col2 = st.columns([2, 1])
with col1:
    court_ids = [c.id for c in courts]
    selected_id = st.selectbox(
        "Court",
        court_ids,
        format_func=lambda cid: find_court(cid).label,
        help="Courts are listed in config/courts.yaml. Edit that file to add or correct one.",
    )
court = find_court(selected_id)
with col2:
    st.metric("Courts here", court.num_courts,
              help="Total courts at this location, from courts.yaml.")

try:
    with st.spinner("Checking the weather"):
        forecast = load_forecast(selected_id)
        scored = load_scored(selected_id)
except WeatherUnavailable as exc:
    st.error(f"Could not load weather data: {exc}")
    st.caption(
        "Scoring needs weather. A stored forecast would normally cover a brief outage, "
        "so this means nothing has been cached yet. Try again with a connection."
    )
    st.stop()

slot_now = recommend.right_now(court, forecast, now)
play, crowd = slot_now.playability, slot_now.crowding

score_bar(play.score, play.label, play.color, play.headline_reason)

c1, c2, c3 = st.columns(3)
c1.metric("Temperature", f"{play.temp_f:.0f}°F")
c2.metric("Wind", f"{play.wind_mph:.0f} mph",
          help="Scored with a lean toward gusts, which ruin a ball toss more than steady wind.")
c3.metric("Courts likely free", f"{crowd.expected_free_courts:.0f} of {court.num_courts}",
          help=f"Crowding estimate: {crowd.label}.")

st.caption(f"Crowding: **{crowd.label}** — {crowd.reason}.")

# When this court is a write-off, point somewhere else rather than dead-ending.
if play.score < threshold:
    with st.spinner("Comparing the other courts"):
        alternatives = {}
        for other in courts:
            if other.id == selected_id:
                continue
            try:
                alternatives[other.id] = recommend.right_now(
                    other, load_forecast(other.id), now
                )
            except WeatherUnavailable:
                continue  # one court failing shouldn't break the comparison
    best_elsewhere = recommend.best_court_right_now(alternatives)
    if best_elsewhere and best_elsewhere.playability.score >= threshold:
        st.info(
            f"**{best_elsewhere.court.label}** is the better bet right now: "
            f"{best_elsewhere.playability.score}/100, "
            f"{best_elsewhere.playability.headline_reason}."
        )
    else:
        st.warning("Every court in the list is scoring poorly. Try Part III for the week ahead.")

with st.expander("Why this score"):
    if play.deductions:
        st.caption("Every hour starts at 100 points. These rules took points off:")
        kv_rows([
            (d.rule, f"blocked — {d.reason}" if d.is_veto
             else f"−{d.points:.0f} — {d.reason}")
            for d in play.deductions
        ])
    else:
        st.caption("No rule deducted anything. Conditions are as good as this model scores.")
    sunrise, sunset = sun_times(court, now.date())
    kv_rows([
        ("Sunrise", format_clock(sunrise)),
        ("Sunset", format_clock(sunset)),
        ("Lights", f"until {court.lights_out}:00" if court.lights else "none at this court"),
        ("Surface", court.surface),
    ])

if not court.verified:
    st.caption(
        f"Details for {court.name} (position, court count, lights) are unconfirmed. "
        "Check them in person and update `config/courts.yaml`."
    )

# ---------------------------------------------------------------------------
# Part II: best times
# ---------------------------------------------------------------------------

st.header("Part II: When should I go?", divider="gray")
st.markdown(
    f"""
Hours are ranked on playability first and crowding second, since bad weather stops play
outright while a crowd only means waiting. Anything scoring under **{threshold}** is left out.
"""
)

tab_today, tab_week = st.tabs(["Today", "Next 7 days"])


def slot_table(slots: list, label_fn) -> None:
    """Render a list of slots as a table with a score bar in each row."""
    frame = pd.DataFrame([
        {
            "When": label_fn(s),
            "Score": s.playability.score,
            "Conditions": s.playability.label,
            "Courts free": s.crowding.expected_free_courts,
            "Crowding": s.crowding.label,
        }
        for s in slots
    ])
    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Playability", format="%d", min_value=0, max_value=100
            ),
            "Courts free": st.column_config.NumberColumn("Courts free", format="%.1f"),
        },
    )


with tab_today:
    today_slots = recommend.best_today(court, forecast, now, min_score=threshold)
    if today_slots:
        best = today_slots[0]
        st.success(f"**Best left today: {format_clock(best.when)}** — {best.summary}")
        slot_table(today_slots[:6], lambda s: format_clock(s.when))
    else:
        st.info("Nothing left today clears the threshold.")

with tab_week:
    week_slots = recommend.best_this_week(
        court, forecast, now, min_score=threshold, per_day=1
    )
    if week_slots:
        overall = max(week_slots, key=lambda s: s.combined_rank)
        st.success(f"**Best of the week: {overall.time_label}** — {overall.summary}")
        st.caption("The best hour on each of the next seven days.")
        slot_table(week_slots, lambda s: s.time_label)
    else:
        st.info("No hour in the next seven days clears the threshold.")

# ---------------------------------------------------------------------------
# Part III: the week ahead
# ---------------------------------------------------------------------------

st.header("Part III: The week ahead", divider="gray")

scope = st.segmented_control(
    "View", ["This court", "Every court"], default="This court"
) or "This court"

if scope == "This court":
    future = scored[
        (scored["time"] >= now.replace(minute=0, second=0, microsecond=0))
        & (scored["time"] <= now + dt.timedelta(days=7))
    ].copy()

    if future.empty:
        st.info("No forecast hours available.")
    else:
        # "%d" rather than "%-d": the no-leading-zero flag is Linux-only and
        # raises on Windows, where this gets developed.
        future["Day"] = future["time"].dt.strftime("%a %d")
        future["Hour"] = future["time"].dt.hour.map(hour_label)
        future = future[(future["time"].dt.hour >= 6) & (future["time"].dt.hour <= 22)]
        future = future.rename(columns={"reason": "Reason"})

        # Days in calendar order; hours fixed 6am to 10pm.
        day_order = future.drop_duplicates("Day").sort_values("time")["Day"].tolist()

        chart = heatmap(
            future, "Day:N", "Hour:N",
            ["Day", "Hour", "score", "Reason"], 520,
            x_sort=day_order, y_sort=HOUR_ORDER,
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"Every hour from 6am to 10pm at {court.name}. Hover a cell for the reason "
            "behind its score. Mornings are at the top."
        )

else:
    with st.spinner("Scoring every court"):
        rows = []
        for other in courts:
            try:
                other_scored = load_scored(other.id)
            except WeatherUnavailable:
                continue
            window = other_scored[
                (other_scored["time"] >= now.replace(minute=0, second=0, microsecond=0))
                & (other_scored["time"] <= now + dt.timedelta(days=2))
            ]
            for _, entry in window.iterrows():
                rows.append({
                    "Court": other.name,
                    "Slot": entry["time"].strftime("%a %H:00"),
                    "time": entry["time"],
                    "score": entry["score"],
                    "Reason": entry["reason"],
                })

    if rows:
        comparison = pd.DataFrame(rows).sort_values("time")
        slot_order = comparison.drop_duplicates("Slot")["Slot"].tolist()
        chart = heatmap(
            comparison, "Slot:N", "Court:N",
            ["Court", "Slot", "score", "Reason"], 320,
            x_sort=slot_order,
        )
        st.altair_chart(chart, width="stretch")
        st.caption("The next 48 hours across every court in courts.yaml.")
    else:
        st.info("No data available.")

# ---------------------------------------------------------------------------
# Part IV: the season
# ---------------------------------------------------------------------------

st.header("Part IV: How much tennis does a Reno year hold?", divider="gray")

years_setting = int(settings.get("data", {}).get("history_years", 3))
st.markdown(
    f"""
The forecast covers this week. This covers the season. Several years of real hourly
weather are scored with the **same rules** used above, then counted up by month, which
is the number worth having when you are planning clinics or arguing for indoor time
in February.
"""
)

# Behind a button because this is the slowest thing here, and Part I should stay quick.
if st.button(f"Load {years_setting} years of history", width="stretch"):
    st.session_state["history_requested"] = True

if st.session_state.get("history_requested"):
    try:
        with st.spinner(f"Fetching and scoring {years_setting} years of weather"):
            monthly = load_history(selected_id, years_setting)
    except WeatherUnavailable as exc:
        st.error(f"Could not load history: {exc}")
        monthly = pd.DataFrame()

    if monthly.empty:
        st.info("No historical data available for this court.")
    else:
        averages = monthly_averages(monthly)

        bars = (
            alt.Chart(averages)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("month_name:N", sort=None, title=None,
                        axis=alt.Axis(labelAngle=0, ticks=False, domain=False)),
                y=alt.Y("playable_hours:Q", title="Playable hours in the month"),
                color=alt.Color(
                    "season:N", title=None,
                    scale=alt.Scale(
                        domain=["Winter", "Spring", "Summer", "Fall"],
                        range=["#2a78d6", "#1baf7a", "#e0a030", "#eb6834"],
                    ),
                    legend=alt.Legend(orient="top", symbolType="square"),
                ),
                tooltip=[
                    alt.Tooltip("month_name:N", title="Month"),
                    alt.Tooltip("playable_hours:Q", title="Playable hours", format=".0f"),
                    alt.Tooltip("playable_pct:Q", title="% of lit hours", format=".0f"),
                    alt.Tooltip("avg_score:Q", title="Average score", format=".0f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(bars, width="stretch")
        st.caption(
            f"Average hours per month at {court.name} that score {threshold} or better, "
            f"over {int(averages['years_counted'].max())} years of recorded weather."
        )

        st.subheader("Is it the weather, or is it the dark?")
        st.markdown(
            "Counting raw hours mixes two different problems together. Measuring instead "
            "against the hours that had **usable light** — daylight, plus floodlights "
            "where a court has them — separates a month that was stormy from a month "
            "that was simply short. One calls for an indoor court, the other for lights."
        )
        line = (
            alt.Chart(averages)
            .mark_line(point=True, strokeWidth=2.5, color="#2a78d6")
            .encode(
                x=alt.X("month_name:N", sort=None, title=None,
                        axis=alt.Axis(labelAngle=0, ticks=False, domain=False)),
                y=alt.Y("playable_pct:Q", title="% of lit hours playable",
                        scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("month_name:N", title="Month"),
                    alt.Tooltip("playable_pct:Q", title="% playable", format=".0f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(line, width="stretch")

        with st.expander("By season, and year by year"):
            st.dataframe(seasonal_summary(monthly), hide_index=True, width="stretch")
            st.dataframe(
                monthly[["year", "month_name", "playable_hours", "daylight_hours",
                         "lit_hours", "avg_score"]],
                hide_index=True, width="stretch", height=320,
                column_config={
                    "year": st.column_config.NumberColumn("Year", format="%d"),
                    "month_name": "Month",
                    "playable_hours": st.column_config.ProgressColumn(
                        "Playable hours", format="%d", min_value=0,
                        max_value=float(monthly["playable_hours"].max()),
                    ),
                    "daylight_hours": st.column_config.NumberColumn("Daylight hrs", format="%d"),
                    "lit_hours": st.column_config.NumberColumn("Lit hrs", format="%d"),
                    "avg_score": st.column_config.NumberColumn("Avg score", format="%.0f"),
                },
            )

# ---------------------------------------------------------------------------
# Part V: how the score works
# ---------------------------------------------------------------------------

st.header("Part V: How the score works", divider="gray")
st.markdown(
    f"""
Two models run side by side, and they stay separate on purpose. They often disagree:
the nicest hour of the week is also the most contested one, while a freezing Tuesday
morning scores badly and guarantees you an empty court. Folding them into one number
would hide exactly that tension.

**Playability** starts every hour at 100 and subtracts using named rules, each of which
reports its own reason:
"""
)
st.dataframe(
    pd.DataFrame([
        {"Rule": "Wet courts", "What it looks at":
         "Rain in the last 8 hours, decayed each hour as the court dries"},
        {"Rule": "Temperature", "What it looks at":
         "Bands from below freezing up to dangerously hot"},
        {"Rule": "Wind", "What it looks at":
         "Bands, weighted 60/40 toward gusts over sustained wind"},
        {"Rule": "Daylight", "What it looks at":
         "Sunrise and sunset for this court's coordinates, plus its lights"},
        {"Rule": "Ice", "What it looks at":
         "Cold combined with recent wet, which either one alone would miss"},
    ]),
    hide_index=True, width="stretch",
)
st.markdown(
    f"""
Some conditions force the score to zero outright rather than merely subtracting: darkness,
active rain, and ice. That separation stops the model reporting a comfortable 68°F at
2am in the rain as a decent bet.

**Crowding** is a separate estimate from day of week, hour, month and the busy hours
recorded per court, nudged by how good the weather is.

Every weight lives in `config/scoring.yaml` with a comment explaining it, so the model
is tuned by editing that file rather than by changing any Python. The current cutoff for
counting an hour as playable is **{threshold}/100**.
"""
)

with st.expander("What this gets wrong"):
    st.markdown(
        """
- **Crowding rests on configured guesses.** It has no way to learn that a tournament
  booked every court this Saturday. Playability is a forecast; crowding is a reminder.
- **The Reno Municipal Tennis Center takes reservations**, which makes its crowding
  estimate the least reliable here. Courts can be booked solid on a day this calls empty.
- **Court details stay unconfirmed** until someone checks them in person, lights especially.
- **Weather is modelled at a grid point** some distance from the court, so a court
  funnelled between buildings runs windier than the forecast says.
- **Drying rates are estimated** rather than measured, making `drying_rate` the largest
  single source of error in the score.
- **Closures, resurfacing, missing nets and pickleball conversions** are all invisible here.
- **Forecast quality falls off with distance.** Day seven deserves much less trust than
  tomorrow, and this app shows that uncertainty nowhere.
        """
    )

report = load_report()
if report:
    with st.expander("Public schedule sources"):
        st.caption(
            f"Last checked {report.get('generated_at', 'unknown')}, with "
            f"{report.get('rate_limit_seconds')} seconds between requests and robots.txt "
            "treated as binding."
        )
        kv_rows([(r["name"], r["reason"]) for r in report.get("results", [])])
        st.caption("Refresh with `python scripts/check_schedules.py`.")

with st.expander("Cached data"):
    for label, status in cache_status().items():
        st.caption(f"{label}: {status}")
    st.caption(
        "Weather responses are stored on disk so repeated runs stay off Open-Meteo's "
        "servers. Forecasts are kept 30 minutes and historical data 30 days."
    )
    if st.button("Clear the cache"):
        clear_cache()
        st.cache_data.clear()
        st.success("Cleared. The next load will fetch fresh data.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built as a student sports analytics project. Scores are estimates from forecast "
    "weather and configured assumptions, and cannot account for closures, bookings, court "
    "repairs or anything else outside the data. Weather from "
    "[Open-Meteo](https://open-meteo.com), licensed "
    "[CC BY 4.0](https://open-meteo.com/en/license). "
    "Sunrise and sunset computed locally with `astral`."
)
