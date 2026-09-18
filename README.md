# Court Conditions

Predicts whether outdoor tennis courts around Reno and Sparks are **playable**
(weather, daylight, court condition) and **free** (crowding), for right now and
for the next seven days.

Built around a question that weather apps answer badly: "can I actually play at
5pm today?" depends on when it last rained, how hard the wind is gusting, and
whether the sun will still be up.

**[Live app](https://court-conditions.streamlit.app)**

---

## What it does

- A current playability score out of 100, with the reason, at the top of the page
- A seven-day hourly heatmap per court
- Best time to play today and this week
- A crowding estimate: roughly how many courts will be free
- A historical view: playable hours per month over the past few years
- Mobile-first layout, since it gets checked standing at the court

---

## Quick start

```bash
git clone https://github.com/sahishnu-m/court-conditions.git
cd court-conditions

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. No API key is required, which is a
deliberate choice explained under *Data sources*.

Run the tests with:

```bash
python -m pytest tests/ -v
```

---

## How the scoring works

Two separate models run side by side, and they stay separate on purpose.

### 1. Playability (0-100)

Every hour starts at **100 points**, and rules subtract from it. Each deduction
carries a plain-English reason, so the app can always say why:

> `62 - courts likely damp: 0.08in of rain, last 2 hours ago (-31); windy
> (17mph, gusting 24) (-20)`

| Rule | What it looks at |
|---|---|
| Wet courts | Rain in the last 8 hours, decayed by an hourly drying rate |
| Temperature | Bands from "below freezing" to "dangerously hot" |
| Wind | Bands, weighted 60/40 toward gusts over sustained wind |
| Daylight and lights | Sunrise and sunset at that court's exact coordinates |
| Freeze and ice | Cold combined with recently wet, which either alone would miss |

**Penalties and vetoes are different things.** A penalty says "this is worse"
and stacks with other penalties. A veto says "this is impossible" and forces
the score to zero: dark, actively raining, or iced over. Without that
distinction the model would happily report "2am, pouring, but a comfortable
68F, so: 74/100".

**The drying curve** is the central idea. A hard court stays unplayable for a
while after rain stops, so the model looks backward in time:

```
penalty = inches x penalty_per_inch x (drying_rate ^ hours_ago)
```

0.10in of rain two hours ago costs `0.10 x 1200 x 0.6^2` = **43 points**. The
same rain five hours ago costs 9, because the court has had time to dry.

### 2. Crowding (0-100)

A separate estimate from day of week, hour, month, and the busy hours you
configure per court, multiplied by a weather response since good conditions
pull people outside.

**Why keep the two apart?** They frequently point in opposite directions. The
nicest hour of the week is also the most contested one, while a freezing
Tuesday morning scores badly and guarantees you an empty court. Merging them
into a single "should I go" number would bury exactly the tension you need to
see.

---

## Tuning it

Every weight lives in `config/scoring.yaml`, heavily commented. Edit the file,
reload the page, done. No Python changes.

The settings most worth correcting from experience:

- `drying_rate` - how fast a Reno hard court sheds water. Raise it toward 0.75
  if the app calls courts dry while they are still puddled.
- `month_multiplier` - the seasonal crowding curve.
- `level` on each `busy_hours` block in `config/courts.yaml`.

**Why rules rather than machine learning?** Training data would mean thousands
of hours hand-labelled as playable or unplayable, and that data has never been
collected. Beyond that, a trained model reporting "43" gives you nothing to
push back on. Here, a wrong call about wet courts traces to one line you can
change. For a model you have to explain and defend, interpretability carries
real weight.

The historical view runs the same rules over past weather, so tuning a weight
moves the history along with the forecast. You can watch a change rewrite the
past, which is a useful check on whether a weight makes sense.

---

## Editing the courts

`config/courts.yaml` holds every court: name, coordinates, court count,
surface, lights, and typical busy hours. Python has none of this hard-coded.

Every court currently reads `verified: false`, meaning nobody has stood on it
and checked. The dashboard flags those. Once you verify one, correct whatever
is wrong and flip the flag.

How far to trust each field:

| Field | Source | Confidence |
|---|---|---|
| `lat` / `lon` | Geocoded from OpenStreetMap | Good, within a block |
| `num_courts` | reno.gov, renotennis.org | Likely right, though courts get converted to pickleball constantly |
| `lights` | Same | Weakest. "Has lights" and "the lights work in November" are different claims |
| `busy_hours` | Guesses | A placeholder for your knowledge, and the field to rewrite first |

To fix coordinates, right-click the courts themselves in Google Maps, rather
than the park entrance, and click the lat/long.

---

## Data sources

**Weather comes from [Open-Meteo](https://open-meteo.com)**, free and with no
API key. That last part is a real design choice: a key would have to be kept
out of git, injected as a secret on Streamlit Cloud, and rotated if it leaked.
A keyless API removes that whole class of problem from a project meant to be
deployed publicly.

- Forecast: current conditions plus seven days hourly
- Archive: several years of hourly history for the seasonal view
- Requested in F, mph and inches directly, so unit conversion appears nowhere
  in the code and unit bugs have nowhere to hide

**Daylight is computed locally** with [`astral`](https://astral.readthedocs.io).
Sunrise and sunset are pure astronomy: given a latitude, longitude and date,
the answer is a fixed calculation. An API call would add a network round trip
and a failure mode in exchange for nothing, and computing it locally keeps this
part working offline.

It matters more than it sounds. Reno sits at about 39.5N, so sunset swings from
roughly 4:45pm in December to 8:30pm in June. That four-hour spread is the
largest seasonal factor in the whole model.

### Caching

API responses are cached to SQLite in `data/` through `requests-cache`, with
two lifetimes, because the two kinds of data behave differently:

- **Forecast: 30 minutes.** A prediction that genuinely changes through the day.
- **Archive: 30 days.** What the weather already was, fixed forever.

One shared lifetime would force a choice between re-downloading years of
history and showing a stale forecast. The cache survives restarts, so quitting
and relaunching re-fetches nothing.

### Scraping public schedules

`scripts/check_schedules.py` optionally checks public city and parks pages for
published court information. The code enforces the rules, leaving nothing to
whoever runs it:

1. **robots.txt is checked first and is binding.** No override flag exists.
   When robots.txt cannot be read at all, the fetch is refused: confirming
   permission is what grants it.
2. **One request every 3 seconds**, enforced inside the fetch method so it
   survives future edits.
3. **Nothing behind a login.** No credentials, no session handling.
4. **Every skip is logged with a reason.**
5. **It identifies itself honestly** in its User-Agent.

**What it found:** very little, reported openly. Washoe County's page fetches
cleanly and mentions no tennis. The Reno and Sparks pages return 404 to our
identified bot while loading fine in a browser, which points to those sites
refusing automated clients.

**That gets left alone.** Spoofing a browser User-Agent would probably retrieve
the pages, but a site blocking bots expresses the same preference as a
robots.txt Disallow through a different mechanism. Honouring robots.txt while
disguising the scraper to defeat bot detection would follow the letter of the
rule while breaking its intent. The app runs fine with no scraped data at all;
this layer is strictly a bonus.

It also stops short of parsing schedules into structured data. Every site has a
different layout, and a parser written against today's HTML would quietly start
producing wrong court hours after the next redesign, which is worse than
producing none. A genuinely structured feed (iCal, JSON, CSV) would be the
point at which a parser earns its place.

---

## Limitations

Worth stating plainly, since this drives real decisions about when to show up:

- **Crowding rests on guesses.** It encodes what you already know as a coach
  and has no way to learn that a tournament booked every court this Saturday.
  Playability is a forecast; crowding is a reminder.
- **The Reno Tennis Center takes reservations** through CourtReserve, making
  its crowding estimate the least reliable in the file. Courts can be booked
  solid on a day this app calls empty.
- **Court details stay unverified** until checked in person, lights especially.
- **Weather is modelled at a grid point** some distance from the court.
  Open-Meteo interpolates, so a court funnelled between two buildings will be
  windier than forecast and a sun-trap court warmer.
- **Drying rates are estimated** rather than measured. This is the largest
  source of error in the playability score.
- **Closures, resurfacing, missing nets and pickleball conversions** are all
  invisible here, and conversions are happening a lot.
- **Forecast quality falls off with distance.** Day seven deserves far less
  trust than tomorrow, and the app currently shows that uncertainty nowhere.

---

## Project layout

```
court-conditions/
├── app.py                      Streamlit dashboard (mobile-first)
├── config/
│   ├── courts.yaml             the courts - edit this
│   └── scoring.yaml            every tunable weight - edit this
├── court_conditions/
│   ├── config.py               loads and validates the YAML
│   ├── cache.py                API response caching
│   ├── weather.py              Open-Meteo forecast and archive
│   ├── daylight.py             local sunrise/sunset maths
│   ├── scoring.py              the playability model
│   ├── crowding.py             the crowding model
│   ├── history.py              playable hours per month
│   ├── recommend.py            "when should I go?"
│   └── scrape.py               polite, optional schedule checking
├── scripts/check_schedules.py  runs the scraper
└── tests/test_scoring.py       28 tests
```

Each module answers one question, so any single file reads top to bottom
without the rest of the app in your head.

---

## Two bugs worth knowing about

Both turned up by running the thing, and both are commented in the code.

**Daylight saving time.** Twice a year local time does something impossible:
1:00am happens twice in November, and 2:00am is skipped entirely in March.
Pandas declines to guess, so `tz_localize` needs explicit `ambiguous` and
`nonexistent` arguments. A seven-day forecast never spans one of those nights,
so it stayed hidden until the first multi-year history pull hit a November.

**A missing cache decorator cost 200x.** `get_timezone()` looked trivially
cheap while re-reading and re-parsing `courts.yaml` from disk on every call,
deep inside the scoring loop. Profiling showed **99% of runtime going to PyYAML
parsing the same file tens of thousands of times**. One `@lru_cache` took the
historical analysis from **226 seconds to 0.8**, with identical output. A
function that looks cheap turns expensive the moment something calls it in a
loop.

---

## License

MIT. Weather data from Open-Meteo under
[CC BY 4.0](https://open-meteo.com/en/license).
