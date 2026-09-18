"""
app.py - Streamlit dashboard. Run with:  streamlit run app.py

LAYOUT NOTES
Built for a phone first, since that's where it gets used - standing at the
court deciding whether to stay.

  * layout="centered" keeps the content column narrow enough to read on a
    phone without side-scrolling.
  * The current verdict sits at the top, visible without scrolling or tapping.
  * Charts are tall rather than wide, matching the shape of a phone screen.
  * Slow work (the multi-year history) sits behind a button so the common
    question stays fast.
  * Court selection lives in the sidebar, which collapses to a menu on mobile.

STREAMLIT'S EXECUTION MODEL (read before editing)
Streamlit re-runs this whole file top to bottom on every widget interaction.
It works by re-execution rather than by callbacks, which is why:
  * @st.cache_data wraps every expensive call. Without it, changing the court
    dropdown would re-download all the weather.
  * The script simply reads current widget values each run and draws.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from court_conditions import recommend
from court_conditions.cache import cache_status, clear_cache
from court_conditions.config import get_timezone, load_courts, load_settings
from court_conditions.daylight import format_clock, sun_times
from court_conditions.history import (
    monthly_averages,
    monthly_playable_hours,
    seasonal_summary,
)
from court_conditions.scoring import playable_threshold, score_series
from court_conditions.scrape import load_report
from court_conditions.weather import WeatherUnavailable, fetch_forecast

st.set_page_config(
    page_title="Court Conditions",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Styling. The palette follows GitHub's dark theme so the app reads as a piece
# of tooling. Colours are defined once here and reused, so retheming is a
# single edit.
st.markdown(
    """
    <style>
      :root {
        --ink:   #e6edf3;
        --muted: #8b949e;
        --line:  #30363d;
      }
      .block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 46rem; }

      .masthead { border-bottom: 1px solid var(--line); padding-bottom: .75rem;
                  margin-bottom: 1.1rem; }
      .masthead h1 { font-size: 1.45rem; font-weight: 600; margin: 0;
                     letter-spacing: -0.01em; }
      .masthead .sub { color: var(--muted); font-size: .85rem; margin-top: .2rem; }

      /* Small chip labels, in the style of repository topic tags. */
      .chips { margin: .1rem 0 1.4rem; }
      .chip { display: inline-block; border: 1px solid var(--line);
              border-radius: 2rem; padding: .12rem .6rem; margin-right: .35rem;
              font-size: .72rem; color: var(--muted); font-family: ui-monospace,
              SFMono-Regular, Menlo, monospace; }

      /* The verdict: a large number over a coloured rule, rather than a filled
         card. Quieter, and easier to read in sunlight. */
      .verdict { padding: .2rem 0 .9rem; }
      .verdict .score { font-size: 3.4rem; font-weight: 600; line-height: 1;
                        letter-spacing: -0.03em; }
      .verdict .outof { font-size: 1rem; color: var(--muted); font-weight: 400; }
      .verdict .state { font-size: 1rem; font-weight: 600; margin-top: .35rem; }
      .verdict .why   { color: var(--muted); font-size: .9rem; margin-top: .15rem; }
      .rule { height: 3px; border-radius: 2px; margin: .85rem 0 0; }

      .section { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em;
                 color: var(--muted); font-weight: 600; margin: 2.2rem 0 .6rem;
                 border-bottom: 1px solid var(--line); padding-bottom: .4rem; }

      .row { display: flex; justify-content: space-between; gap: 1rem;
             padding: .45rem 0; border-bottom: 1px solid var(--line);
             font-size: .88rem; }
      .row .k { color: var(--ink); white-space: nowrap; }
      .row .v { color: var(--muted); text-align: right; }

      .note { color: var(--muted); font-size: .82rem; line-height: 1.5; }

      [data-testid="stMetricValue"] { font-size: 1.25rem; font-weight: 600; }
      [data-testid="stMetricLabel"] { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

TIMEZONE = ZoneInfo(get_timezone())

# One shared colour scale, matching the label colours in scoring.yaml.
SCALE = [
    [0.0, "#f85149"],
    [0.4, "#db6d28"],
    [0.6, "#d29922"],
    [0.8, "#57ab5a"],
    [1.0, "#3fb950"],
]

CHART_FONT = dict(size=11, color="#8b949e")
TRANSPARENT = "rgba(0,0,0,0)"


# ---------------------------------------------------------------------------
# Cached loaders.
#
# @st.cache_data keys on the arguments, so these take plain strings that hash
# predictably rather than Court objects.
#
# This sits on top of the requests_cache layer in cache.py. The two do
# different jobs: requests_cache skips the network call, this skips re-parsing
# and re-scoring on every widget interaction.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast_for(court_id: str) -> pd.DataFrame:
    """Forecast for one court. Cached 30 minutes."""
    from court_conditions.config import find_court

    return fetch_forecast(find_court(court_id))


@st.cache_data(ttl=1800, show_spinner=False)
def scored_forecast_for(court_id: str) -> pd.DataFrame:
    """Forecast run through the playability rules. Cached 30 minutes."""
    from court_conditions.config import find_court

    return score_series(find_court(court_id), load_forecast_for(court_id))


@st.cache_data(ttl=86400, show_spinner=False)
def history_for(court_id: str, years: int) -> pd.DataFrame:
    """
    Multi-year historical analysis. Cached for a day, since it is the slowest
    operation here and the answer moves very slowly.
    """
    from court_conditions.config import find_court

    return monthly_playable_hours(find_court(court_id), years=years)


def row(key: str, value: str) -> None:
    """One key/value line with a hairline rule under it."""
    st.markdown(
        f'<div class="row"><span class="k">{key}</span>'
        f'<span class="v">{value}</span></div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    """A small uppercase section heading."""
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def style_chart(figure, height: int) -> None:
    """Apply the shared dark chart styling in one place."""
    figure.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=CHART_FONT,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
courts = load_courts()
settings = load_settings()

with st.sidebar:
    st.markdown("**Settings**")

    court_labels = {c.id: c.label for c in courts}
    selected_id = st.selectbox(
        "Court",
        options=list(court_labels.keys()),
        format_func=lambda cid: court_labels[cid],
        help="Add, remove or correct courts in config/courts.yaml.",
    )

    st.divider()
    st.caption("Cached data")
    for label, status in cache_status().items():
        st.caption(f"{label}: {status}")
    if st.button("Clear cache", width="stretch"):
        clear_cache()
        st.cache_data.clear()
        st.success("Cleared. The next load will re-fetch.")

    st.divider()
    st.caption(
        "Scoring weights live in config/scoring.yaml. Edit that file and "
        "reload this page to see the change."
    )

selected_court = next(c for c in courts if c.id == selected_id)
now = dt.datetime.now(TIMEZONE)
threshold = playable_threshold()


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
lights_chip = (
    f"lights to {selected_court.lights_out}:00"
    if selected_court.lights
    else "no lights"
)

st.markdown(
    f"""
    <div class="masthead">
      <h1>Court Conditions</h1>
      <div class="sub">{selected_court.label} &nbsp;&middot;&nbsp;
        {now.strftime('%A %d %B')}, {format_clock(now)}</div>
    </div>
    <div class="chips">
      <span class="chip">{selected_court.num_courts} courts</span>
      <span class="chip">{selected_court.surface}</span>
      <span class="chip">{lights_chip}</span>
      <span class="chip">{'verified' if selected_court.verified else 'unverified'}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Forecast load. Everything below depends on it, so a failure here stops the
# page with an explanation instead of a stack trace.
# ---------------------------------------------------------------------------
try:
    with st.spinner("Loading weather"):
        forecast = load_forecast_for(selected_id)
        scored = scored_forecast_for(selected_id)
except WeatherUnavailable as exc:
    st.error(f"Could not load weather data: {exc}")
    st.caption(
        "Scoring needs weather. A cached forecast would normally cover a brief "
        "outage, which means there isn't one stored yet. Try again once you "
        "have a connection."
    )
    st.stop()


# ===========================================================================
# Current conditions
# ===========================================================================
slot_now = recommend.right_now(selected_court, forecast, now)
play = slot_now.playability
crowd = slot_now.crowding

st.markdown(
    f"""
    <div class="verdict">
      <div class="score">{play.score}<span class="outof"> / 100</span></div>
      <div class="state" style="color:{play.color};">{play.label}</div>
      <div class="why">{play.headline_reason}</div>
      <div class="rule" style="background:{play.color};"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Temperature", f"{play.temp_f:.0f}°F")
col_b.metric("Wind", f"{play.wind_mph:.0f} mph")
col_c.metric(
    "Courts free", f"{crowd.expected_free_courts:.0f} of {selected_court.num_courts}"
)

st.markdown(
    f'<div class="note">Crowding: {crowd.label}. {crowd.reason}.</div>',
    unsafe_allow_html=True,
)

# When this court is a write-off, point somewhere else instead of dead-ending.
if play.score < threshold:
    with st.spinner("Comparing the other courts"):
        alternatives: dict[str, recommend.Slot] = {}
        for other in courts:
            if other.id == selected_id:
                continue
            try:
                alternatives[other.id] = recommend.right_now(
                    other, load_forecast_for(other.id), now
                )
            except WeatherUnavailable:
                continue  # one court failing shouldn't break the comparison

    best_elsewhere = recommend.best_court_right_now(alternatives)
    if best_elsewhere and best_elsewhere.playability.score >= threshold:
        st.info(
            f"{best_elsewhere.court.label} is better right now: "
            f"{best_elsewhere.playability.score}/100, "
            f"{best_elsewhere.playability.headline_reason}."
        )
    else:
        st.info("Every court in the list is scoring poorly. See the week view below.")

with st.expander("Score breakdown"):
    if play.deductions:
        st.caption("Started at 100 points.")
        for deduction in play.deductions:
            if deduction.is_veto:
                row(deduction.rule, f"blocked - {deduction.reason}")
            else:
                row(deduction.rule, f"-{deduction.points:.0f} - {deduction.reason}")
    else:
        st.caption("No deductions were applied.")

    sunrise, sunset = sun_times(selected_court, now.date())
    row("sunrise", format_clock(sunrise))
    row("sunset", format_clock(sunset))

if not selected_court.verified:
    st.caption(
        f"Details for {selected_court.name} (position, court count, lights) are "
        "unconfirmed. Check them in person and update config/courts.yaml."
    )


# ===========================================================================
# Best times
# ===========================================================================
section("Best times")

tab_today, tab_week = st.tabs(["Today", "Next 7 days"])

with tab_today:
    today_slots = recommend.best_today(
        selected_court, forecast, now, min_score=threshold
    )
    if today_slots:
        best = today_slots[0]
        st.markdown(f"**{format_clock(best.when)}** &nbsp;{best.summary}")
        for slot in today_slots[1:5]:
            row(format_clock(slot.when), slot.summary)
    else:
        st.caption("Nothing left today clears the threshold.")

with tab_week:
    week_slots = recommend.best_this_week(
        selected_court, forecast, now, min_score=threshold, per_day=1
    )
    if week_slots:
        overall_best = max(week_slots, key=lambda s: s.combined_rank)
        st.markdown(
            f"**Best of the week: {overall_best.time_label}** &nbsp;{overall_best.summary}"
        )
        st.caption("Best hour each day")
        for slot in week_slots:
            row(slot.time_label, slot.summary)
    else:
        st.caption("No hour in the next 7 days clears the threshold.")


# ===========================================================================
# Heatmap
# ===========================================================================
section("Hourly outlook")

heatmap_scope = st.radio(
    "Scope",
    ["This court", "All courts"],
    horizontal=True,
    label_visibility="collapsed",
)

if heatmap_scope == "This court":
    # Hours on the vertical axis, days across the top. The other orientation
    # would need 24 columns, which on a phone means unreadable slivers.
    future = scored[
        (scored["time"] >= now.replace(minute=0, second=0, microsecond=0))
        & (scored["time"] <= now + dt.timedelta(days=7))
    ]

    if future.empty:
        st.caption("No forecast hours available.")
    else:
        grid = future.copy()
        # "%d" rather than "%-d": the no-leading-zero flag is Linux-only and
        # raises on Windows. Same portability trap as format_clock().
        grid["day"] = grid["time"].dt.strftime("%a %d")
        grid["hour"] = grid["time"].dt.hour

        day_order = grid.drop_duplicates("day").sort_values("time")["day"].tolist()
        pivot = grid.pivot_table(
            index="hour", columns="day", values="score", aggfunc="first"
        ).reindex(columns=day_order)
        reasons = grid.pivot_table(
            index="hour", columns="day", values="reason", aggfunc="first"
        ).reindex(columns=day_order)

        # Trim to hours anyone would consider playing.
        pivot = pivot[(pivot.index >= 6) & (pivot.index <= 22)]
        reasons = reasons.reindex(pivot.index)

        figure = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=[f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in pivot.index],
                customdata=reasons.values,
                hovertemplate="%{x} %{y}<br>Score %{z}<br>%{customdata}<extra></extra>",
                colorscale=SCALE,
                zmin=0,
                zmax=100,
                xgap=2,
                ygap=2,
                colorbar=dict(thickness=10, outlinewidth=0, title=""),
            )
        )
        style_chart(figure, 540)
        figure.update_layout(yaxis=dict(autorange="reversed"))  # morning at top
        st.plotly_chart(figure, width="stretch")

else:
    with st.spinner("Scoring every court"):
        comparison_rows = []
        for court in courts:
            try:
                court_scored = scored_forecast_for(court.id)
            except WeatherUnavailable:
                continue
            window = court_scored[
                (court_scored["time"] >= now.replace(minute=0, second=0, microsecond=0))
                & (court_scored["time"] <= now + dt.timedelta(days=2))
            ]
            for _, entry in window.iterrows():
                comparison_rows.append(
                    {"court": court.name, "when": entry["time"], "score": entry["score"]}
                )

    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows)
        comparison["slot"] = comparison["when"].dt.strftime("%a %H:00")
        slot_order = (
            comparison.drop_duplicates("slot").sort_values("when")["slot"].tolist()
        )
        pivot = comparison.pivot_table(
            index="court", columns="slot", values="score", aggfunc="first"
        ).reindex(columns=slot_order)

        figure = px.imshow(
            pivot, color_continuous_scale=SCALE, zmin=0, zmax=100, aspect="auto"
        )
        style_chart(figure, 400)
        figure.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            coloraxis_colorbar=dict(thickness=10, outlinewidth=0, title=""),
        )
        st.plotly_chart(figure, width="stretch")
        st.markdown(
            '<div class="note">Next 48 hours across every court in courts.yaml.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No data available.")


# ===========================================================================
# History
# ===========================================================================
section("Historical playable hours")

years_setting = int(settings.get("data", {}).get("history_years", 3))

st.markdown(
    '<div class="note">Past weather run through these same scoring rules. '
    "Tuning a weight in scoring.yaml moves these numbers too.</div>",
    unsafe_allow_html=True,
)

# Behind a button because it is the slowest operation here, and the question at
# the top of the page should stay fast.
if st.button(f"Load {years_setting} years of history", width="stretch"):
    st.session_state["history_requested"] = True

if st.session_state.get("history_requested"):
    try:
        with st.spinner(f"Fetching and scoring {years_setting} years"):
            monthly = history_for(selected_id, years_setting)
    except WeatherUnavailable as exc:
        st.error(f"Could not load history: {exc}")
        monthly = pd.DataFrame()

    if monthly.empty:
        st.caption("No historical data available for this court.")
    else:
        averages = monthly_averages(monthly)

        figure = px.bar(
            averages,
            x="month_name",
            y="playable_hours",
            color="season",
            color_discrete_map={
                "Winter": "#388bfd",
                "Spring": "#3fb950",
                "Summer": "#d29922",
                "Fall": "#db6d28",
            },
            labels={"month_name": "", "playable_hours": "Playable hours"},
            hover_data={"playable_pct": ":.0f", "avg_score": ":.0f"},
        )
        style_chart(figure, 380)
        figure.update_layout(legend=dict(orientation="h", y=-0.18, title=""))
        figure.update_xaxes(showgrid=False)
        figure.update_yaxes(gridcolor="#21262d")
        st.plotly_chart(figure, width="stretch")

        st.markdown(
            f'<div class="note">Average playable hours per month at '
            f"{selected_court.name}, counting hours that score {threshold} or "
            f'above, over {int(averages["years_counted"].max())} years.</div>',
            unsafe_allow_html=True,
        )

        with st.expander("By season"):
            st.dataframe(seasonal_summary(monthly), width="stretch", hide_index=True)

        with st.expander("Share of usable light"):
            st.caption(
                "Measured against hours with usable light: daylight, plus lights "
                "where the court has them. Dividing by all 24 hours would make "
                "December look bad for a reason that comes down to darkness."
            )
            pct = px.line(averages, x="month_name", y="playable_pct", markers=True)
            pct.update_traces(line_color="#3fb950")
            style_chart(pct, 300)
            pct.update_layout(xaxis_title=None, yaxis_title="% of lit hours")
            pct.update_yaxes(range=[0, 100], gridcolor="#21262d")
            pct.update_xaxes(showgrid=False)
            st.plotly_chart(pct, width="stretch")

        with st.expander("Year by year"):
            st.dataframe(
                monthly[
                    [
                        "year", "month_name", "playable_hours",
                        "daylight_hours", "lit_hours", "avg_score",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )


# ===========================================================================
# Method and limits
# ===========================================================================
section("Method")

st.markdown(
    f"""
**Two separate models, kept separate on purpose.**

*Playability* (0-100) starts at 100 and subtracts using rules you can read and
edit in `config/scoring.yaml`: recent rain with a drying curve, temperature
bands, wind weighted toward gusts, daylight and lights, and a freeze check.
Some conditions force the score to zero outright, covering dark, actively
raining, and iced over.

*Crowding* (0-100) is a separate estimate built from day of week, hour, month,
and the busy hours recorded in `config/courts.yaml`.

**Known limits**

- Crowding rests on configured guesses, so a tournament booking every court on
  a Saturday will pass it by. Playability is a forecast; crowding is a reminder.
- Court details stay unconfirmed until someone checks them in person. Anything
  still marked unverified carries a note.
- Weather is modelled at a grid point some distance from the court. A court
  funnelled between buildings will be windier than the forecast says.
- Drying rates are estimates. `drying_rate` in scoring.yaml is the setting most
  worth correcting once you have watched a few courts dry out.
- Closures, resurfacing, missing nets and courts converted to pickleball are
  all invisible here.

Weather from [Open-Meteo](https://open-meteo.com). Sunrise and sunset are
computed locally with `astral`, which keeps that part working offline.

Playable threshold: {threshold}/100.
    """
)

report = load_report()
if report:
    with st.expander("Public schedule sources"):
        st.caption(
            f"Checked {report.get('generated_at', 'unknown')}, "
            f"{report.get('rate_limit_seconds')}s between requests, "
            "robots.txt honoured."
        )
        for result in report.get("results", []):
            row(result["name"], result["reason"])
        st.caption("Refresh with: python scripts/check_schedules.py")

st.markdown(
    '<div class="note" style="margin-top:2rem;">All times are Reno local. '
    "Weather data from Open-Meteo.</div>",
    unsafe_allow_html=True,
)
