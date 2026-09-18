"""
court_conditions — predicts whether Reno-area outdoor tennis courts are
playable and free at a given day and hour.

The package is split so each file answers one question:

    config.py     What courts exist and what are the tuning numbers?
    cache.py      How do we avoid hammering the weather API?
    weather.py    What is / was the weather at this court?
    daylight.py   Is the sun up at this court at this hour?
    scoring.py    Is the court PLAYABLE?   (0-100, rule-based)
    crowding.py   Is the court FREE?       (0-100, rule-based)
    history.py    How many playable hours did each month have?
    scrape.py     Are any court schedules published publicly?

Keeping them separate means you can open one file, read it top to bottom, and
understand that one idea without holding the rest of the app in your head.
"""

__version__ = "1.0.0"
