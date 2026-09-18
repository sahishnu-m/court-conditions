"""
config.py — loads courts.yaml and scoring.yaml into Python objects.

DESIGN CHOICE: why dataclasses instead of just passing the raw dictionaries
around?

If the rest of the app used raw dicts, a typo like `court["ligths"]` would
blow up at runtime, deep inside the dashboard, with a confusing error. By
converting the YAML into dataclasses here, a bad field name fails immediately
at startup with a clear message, and your editor can autocomplete `court.lights`.
This file is the ONLY place that knows what the YAML looks like — if you add a
field to courts.yaml, this is the one Python file you have to touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# The project root is the folder that contains this package's parent directory.
# __file__ is this file, .parent is court_conditions/, .parent again is the repo
# root. Deriving it this way means the app works no matter what directory you
# happen to run `streamlit run app.py` from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Day-of-week names used in courts.yaml, in Python's own Monday=0 order so we
# can index straight into this list with `datetime.weekday()`.
WEEKDAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass(frozen=True)
class BusyBlock:
    """One 'this place is busy at these times' entry from courts.yaml."""

    days: tuple[str, ...]  # e.g. ("mon", "tue", "wed", "thu", "fri")
    start_hour: int  # inclusive, 24-hour clock
    end_hour: int  # EXCLUSIVE — "17-20" means 17, 18, 19 but not 20
    level: float  # 0.0 to 1.0, how full it gets
    why: str = ""  # your note about where the number came from

    def applies_to(self, weekday_index: int, hour: int) -> bool:
        """True if this block covers the given weekday (Mon=0) and hour."""
        return (
            WEEKDAY_NAMES[weekday_index] in self.days
            and self.start_hour <= hour < self.end_hour
        )


@dataclass(frozen=True)
class Court:
    """A single tennis location."""

    id: str
    name: str
    city: str
    lat: float
    lon: float
    num_courts: int
    surface: str
    lights: bool
    lights_out: int | None  # hour the lights shut off; None when lights is False
    verified: bool
    notes: str = ""
    busy_hours: tuple[BusyBlock, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """Name as shown in the UI, with the city so Deer Park isn't ambiguous."""
        return f"{self.name} ({self.city})"


def _parse_hour_range(text: str) -> tuple[int, int]:
    """
    Turn the string "17-20" into the pair (17, 20).

    Kept as a tiny separate function so the error message can be specific:
    a typo in courts.yaml should tell you exactly which value was wrong.
    """
    try:
        start_text, end_text = str(text).split("-")
        return int(start_text), int(end_text)
    except ValueError as exc:
        raise ValueError(
            f"Bad hours value {text!r} in courts.yaml. "
            'Expected something like "17-20" (5pm to 8pm).'
        ) from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file, with a friendly error if it's missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Both config/courts.yaml and "
            f"config/scoring.yaml must exist next to the app."
        )
    # safe_load (not load) because safe_load cannot execute arbitrary Python
    # embedded in a YAML file. It costs nothing and is simply the correct
    # default for reading a config file.
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_courts() -> tuple[Court, ...]:
    """
    Load and validate config/courts.yaml.

    @lru_cache means the file is read from disk once per process and the
    parsed result is reused. Streamlit re-runs the whole script on every
    interaction, so without this we'd re-read and re-parse the YAML every
    time you moved a slider.
    """
    raw = _load_yaml(CONFIG_DIR / "courts.yaml")
    courts: list[Court] = []

    for entry in raw.get("courts", []):
        # Build the busy-hour blocks first so a bad one names its court.
        blocks: list[BusyBlock] = []
        for block in entry.get("busy_hours", []) or []:
            start, end = _parse_hour_range(block["hours"])
            blocks.append(
                BusyBlock(
                    days=tuple(d.lower() for d in block["days"]),
                    start_hour=start,
                    end_hour=end,
                    level=float(block.get("level", 0.5)),
                    why=block.get("why", ""),
                )
            )

        has_lights = bool(entry.get("lights", False))
        courts.append(
            Court(
                id=entry["id"],
                name=entry["name"],
                city=entry.get("city", ""),
                lat=float(entry["lat"]),
                lon=float(entry["lon"]),
                num_courts=int(entry.get("num_courts", 1)),
                surface=str(entry.get("surface", "hard")).lower(),
                lights=has_lights,
                # A lights_out hour is meaningless without lights, so we force
                # it to None rather than trusting whatever the YAML said.
                lights_out=int(entry["lights_out"])
                if has_lights and entry.get("lights_out") is not None
                else None,
                verified=bool(entry.get("verified", False)),
                notes=(entry.get("notes") or "").strip(),
                busy_hours=tuple(blocks),
            )
        )

    if not courts:
        raise ValueError("courts.yaml contained no courts — nothing to predict.")

    # Catch the copy-paste mistake of duplicating a court block without
    # changing its id, which would silently make one court shadow the other.
    ids = [c.id for c in courts]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate court id(s) in courts.yaml: {sorted(duplicates)}")

    return tuple(courts)


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """
    Load config/scoring.yaml.

    DESIGN CHOICE: this one stays a plain nested dictionary rather than
    becoming dataclasses. The scoring config is read in a handful of places
    that each want a different slice of it, and its shape changes every time
    you add a tuning knob. Freezing that shape into dataclasses would mean
    editing Python every time you wanted a new number — exactly the friction
    this project is trying to avoid.
    """
    return _load_yaml(CONFIG_DIR / "scoring.yaml")


@lru_cache(maxsize=1)
def get_timezone() -> str:
    """
    The IANA timezone name all times in this app are expressed in.

    The @lru_cache here carries real weight: leaving it off made the
    historical analysis take over three minutes.

    This function is called deep inside the scoring loop (every hour asks for
    sunrise/sunset, which needs the timezone). Without the cache, every one of
    those calls re-opened courts.yaml and re-parsed the whole file. Profiling
    showed 99% of the total runtime was PyYAML parsing the same file tens of
    thousands of times.

    The lesson worth keeping: a function that looks trivially cheap becomes
    expensive when something calls it in a loop. The right fix is to make the
    cheap thing genuinely cheap, so every caller benefits at once.
    """
    raw = _load_yaml(CONFIG_DIR / "courts.yaml")
    return raw.get("meta", {}).get("timezone", "America/Los_Angeles")


def find_court(court_id: str) -> Court:
    """Look up one court by id, with a helpful error listing the valid ids."""
    for court in load_courts():
        if court.id == court_id:
            return court
    known = ", ".join(c.id for c in load_courts())
    raise KeyError(f"No court with id {court_id!r}. Known ids: {known}")
