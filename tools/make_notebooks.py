"""Write the five issue-reproduction notebooks in examples/.

    python tools/make_notebooks.py            # write (no outputs)
    python tools/make_notebooks.py --run      # write, execute and save the outputs (needs nbclient + ipykernel)
    python tools/make_notebooks.py --test     # execute the committed notebooks without writing (CI)

Each notebook shows the leak with the code shape the report describes, then the fin-pit fix. Values
the reports do not give are synthetic and labelled as such.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "examples")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "tradingagents_1275_fred_vintages.ipynb": [
        md("""
# TradingAgents #1275: FRED serves today's vintage to a historical run

https://github.com/TauricResearch/TradingAgents/issues/1275

`get_macro_data` bounded FRED requests by observation date only. The vintage window
(`realtime_start`/`realtime_end`) defaults to today, so a run dated 2023-03-31 received March CPI,
released 2023-04-12, at its revised value.

The frame below has the shape of `fredapi.Fred.get_series_all_releases("CPIAUCSL")`. Release
dates for the February and March 2023 prints are the real ones; the values are synthetic.
"""),
        code("""
import pandas as pd
import fin_pit as pit
from fin_pit.adapters import fred

raw = pd.DataFrame({
    "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-01-01", "2023-02-01", "2023-03-01"]),
    "realtime_start": pd.to_datetime(["2023-02-14", "2023-03-14", "2023-04-12", "2024-02-13", "2024-02-13", "2024-02-13"]),
    "value": [299.2, 300.8, 301.8, 299.0, 300.5, 301.6],   # synthetic
})
"""),
        md("## The leak: observation-date bound, today's vintage"),
        code("""
latest = raw.sort_values("realtime_start").groupby("date").tail(1)
latest[latest["date"] <= "2023-03-31"].sort_values("date")
"""),
        md("## With fin-pit: vintages stamped with their release, read as of the decision"),
        code("""
stamped = fred.from_fredapi(raw, "CPIAUCSL")
with pit.asof("2023-03-31 16:00", tz="America/New_York"):
    cpi = fred.asof_series(stamped)
cpi
"""),
        code("""
assert list(cpi.index.strftime("%Y-%m-%d")) == ["2023-01-01", "2023-02-01"]
assert cpi.tolist() == [299.2, 300.8]
fred.observation_params(pit.to_utc("2023-03-31 16:00", "America/New_York"))   # for raw API calls
"""),
    ],
    "tradingagents_1300_live_snapshot.ipynb": [
        md("""
# TradingAgents #1300: a live snapshot inside a 2024 analysis

https://github.com/TauricResearch/TradingAgents/issues/1300

`get_fundamentals` stamped its header with `datetime.now()`. For a symbol with no statements the
whole payload was a market snapshot dated 2026-09-04 inside a run dated 2024-05-10.
"""),
        code("""
import fin_pit as pit
from fin_pit.adapters import yfinance as yfa

info = {"marketCap": 1_200_000_000_000}   # synthetic

def get_fundamentals(info):
    return f"# Data retrieved on: 2026-09-04\\nmarketCap: {info['marketCap']}"

print(get_fundamentals(info))   # nothing stops it
"""),
        md("## With fin-pit"),
        code("""
guarded = pit.guard(get_fundamentals, scan_text=True)
snapshot = pit.guard(lambda: yfa.snapshot(info, retrieved_at="2026-09-04T14:00Z"))

with pit.asof("2024-05-10 16:00", tz="America/New_York"):
    for fn, args in ((guarded, (info,)), (snapshot, ())):
        try:
            fn(*args)
        except pit.LookaheadError as e:
            print("refused:", e)
"""),
    ],
    "tradingagents_1518_after_close.ipynb": [
        md("""
# TradingAgents #1518: news and filings after the close

https://github.com/TauricResearch/TradingAgents/issues/1518

A decision dated d kept news until 00:00 UTC of d+1 (20:00 New York in summer) and compared
EDGAR `filed` dates with d, while the decision was scored from the 16:00 close of d. The report's
example: Nike's earnings 8-K accepted 16:15 New York (20:15 UTC) on 2026-10-01, CNBC at
20:07 UTC, Yahoo Finance at 20:19 UTC. The accession number below is a placeholder.
"""),
        code("""
import datetime as dt
import pandas as pd
import fin_pit as pit
from fin_pit.adapters import edgar

news = [
    {"title": "midday preview", "available_at": "2026-10-01T17:00:00Z"},
    {"title": "Nike shares sink in extended trading", "available_at": "2026-10-01T20:07:00Z"},
    {"title": "Nike reports sales below estimates", "available_at": "2026-10-01T20:19:00Z"},
]
submissions = {"filings": {"recent": {
    "accessionNumber": ["0000320187-26-000099"], "filingDate": ["2026-10-01"], "reportDate": ["2026-10-01"],
    "acceptanceDateTime": ["2026-10-01T20:15:00.000Z"], "form": ["8-K"], "items": ["2.02,9.01"],
    "primaryDocument": ["nke-8k.htm"]}}}
filings = edgar.filings(submissions)
"""),
        md("## The leak: calendar-day windows"),
        code("""
day_start = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
print([n["title"] for n in news if day_start <= pit.to_utc(n["available_at"]) < day_start + dt.timedelta(days=1)])
print(filings[pd.to_datetime(filings["filing_date"]).dt.date <= dt.date(2026, 10, 1)][["form", "items", "available_at"]])
"""),
        md("## With fin-pit: cut at the decision time, the 16:00 close"),
        code("""
with pit.asof("2026-10-01 16:00", tz="America/New_York"):
    print([n["title"] for n in pit.filter(news)])
    print(pit.filter(filings).empty)
"""),
    ],
    "tradingagents_1679_short_history.ipynb": [
        md("""
# TradingAgents #1679: a 200-day SMA from 45 sessions

https://github.com/TauricResearch/TradingAgents/issues/1679

stockstats computes rolling windows with `min_periods=1`, so on about 45 sessions of history
`close_50_sma` and `close_200_sma` are the same 45-bar mean. The prices here are synthetic.
"""),
        code("""
import numpy as np
import pandas as pd
import fin_pit as pit

rng = np.random.default_rng(1679)
df = pd.DataFrame({"close": 150 * np.exp(np.cumsum(rng.normal(0, 0.015, 45)))})
df["close"].rolling(50, min_periods=1).mean().iloc[-1], df["close"].rolling(200, min_periods=1).mean().iloc[-1]
"""),
        md("## With fin-pit"),
        code("""
df.pit.indicator("sma", 200).tail(3)
"""),
    ],
    "finrl_1463_backfilled_fundamentals.ipynb": [
        md("""
# FinRL #1463: fundamentals keyed on the period end, back-filled across tickers

https://github.com/AI4Finance-Foundation/FinRL/issues/1463

The report's minimal reproduction: ratios keyed on Compustat `datadate` (fiscal period end), then
`bfill` over the whole frame. Each quarter's EPS shows up before the quarter ends, and the last
ticker's tail is filled from the next ticker.
"""),
        code("""
import datetime as dt, itertools
import pandas as pd
import fin_pit as pit

dates = pd.date_range("2020-01-01", "2020-07-10")
ratios = pd.DataFrame({"date": pd.to_datetime(["2020-03-31", "2020-06-30"] * 2),
                       "tic": ["AAA", "AAA", "BBB", "BBB"], "EPS": [1.0, 2.0, 10.0, 20.0]})
grid = pd.DataFrame(list(itertools.product(dates, ["AAA", "BBB"])), columns=["date", "tic"])
full = grid.merge(ratios, how="left", on=["date", "tic"]).sort_values(["tic", "date"]).bfill(axis="rows")
full[(full.tic == "AAA") & full.date.isin(pd.to_datetime(["2020-01-02", "2020-04-01", "2020-07-05"]))]
"""),
        md("""
## With fin-pit: join on availability, per ticker

Key each quarter on its report date (Compustat `rdq`; here synthetic dates about a month after the
period end), available at the end of that day in New York, and join each trading day's 16:00
decision to the newest report known by then.
"""),
        code("""
ratios["rdq"] = pd.to_datetime(["2020-04-28", "2020-07-28", "2020-04-30", "2020-07-30"])   # synthetic
ratios["available_at"] = [pit.to_utc(dt.datetime.combine(d.date(), dt.time(23, 59, 59)), "America/New_York") for d in ratios["rdq"]]
grid["decision_time"] = [pit.to_utc(dt.datetime.combine(d.date(), dt.time(16)), "America/New_York") for d in grid["date"]]
pit_full = pit.asof_join(grid, ratios.drop(columns=["date"]), "decision_time", by="tic")
pit_full[(pit_full.tic == "AAA") & pit_full.date.isin(pd.to_datetime(["2020-01-02", "2020-04-01", "2020-04-29", "2020-07-05"]))][["date", "tic", "EPS"]]
"""),
    ],
}


def execute(path, save):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": OUT}}).execute()
    if save:
        for cell in nb.cells:
            cell.get("metadata", {}).pop("execution", None)
        nbformat.write(nb, path)
    return nb


def main():
    if "--test" in sys.argv:
        for name in NOTEBOOKS:
            execute(os.path.join(OUT, name), save=False)
            print("ok", name)
        return
    os.makedirs(OUT, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(notebook(cells), fh, indent=1)
            fh.write("\n")
    print(f"wrote {len(NOTEBOOKS)} notebooks to {OUT}")
    if "--run" in sys.argv:
        for name in NOTEBOOKS:
            execute(os.path.join(OUT, name), save=True)
            print("ran", name)


if __name__ == "__main__":
    main()
