"""One golden test per real look-ahead report. Each test first shows the leak with the code
shape the issue describes (the "without" half), then shows fin-pit stopping it.

* TradingAgents #1275: FRED observations bounded by observation date, served from today's vintage.
* TradingAgents #1300: a live fundamentals snapshot stamped with now() inside a historical run.
* TradingAgents #1518: news and EDGAR filings released after the 16:00 close admitted to a
  decision executed at that close.
* TradingAgents #1679: 50 and 200 bar SMAs computed from ~45 sessions.
* FinRL #1463: quarterly ratios keyed on the fiscal period end and back-filled across tickers.
"""
import datetime as dt
import itertools
import os
import json

import numpy as np
import pandas as pd
import pytest

import fin_pit as pit
from fin_pit.adapters import edgar, fred, yfinance as yfa

DATA = os.path.join(os.path.dirname(__file__), "data")


# --- TradingAgents #1275 -------------------------------------------------------------------


def all_releases():
    # fredapi.Fred.get_series_all_releases("CPIAUCSL") shape. March 2023 CPI was released
    # 2023-04-12 (the date in the issue); the values here are synthetic.
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-01-01", "2023-02-01", "2023-03-01"]),
            "realtime_start": pd.to_datetime(["2023-02-14", "2023-03-14", "2023-04-12", "2024-02-13", "2024-02-13", "2024-02-13"]),
            "value": [299.2, 300.8, 301.8, 299.0, 300.5, 301.6],
        }
    )


def get_macro_data_like_the_issue(raw, curr_date):
    """observation_end = curr_date, vintage = today (the issue's call shape)."""
    latest = raw.sort_values("realtime_start").groupby("date").tail(1)
    return latest[latest["date"] <= pd.Timestamp(curr_date)].sort_values("date")


def test_tradingagents_1275_fred_vintages():
    raw = all_releases()
    leaked = get_macro_data_like_the_issue(raw, "2023-03-31")
    assert pd.Timestamp("2023-03-01") in set(leaked["date"])  # released 2023-04-12
    assert leaked["value"].tolist()[:2] == [299.0, 300.5]  # 2024 revisions

    stamped = fred.from_fredapi(raw, "CPIAUCSL")
    with pit.asof("2023-03-31 16:00", tz="America/New_York"):
        s = fred.asof_series(stamped)
    assert s.index.strftime("%Y-%m-%d").tolist() == ["2023-01-01", "2023-02-01"]
    assert s.tolist() == [299.2, 300.8]  # the values as first published
    assert fred.observation_params(pit.to_utc("2023-03-31 16:00", "America/New_York")) == {
        "realtime_start": "2023-03-30",
        "realtime_end": "2023-03-30",
    }


# --- TradingAgents #1300 -------------------------------------------------------------------


def get_fundamentals_like_the_issue(info):
    return f"# Data retrieved on: {dt.date(2026, 9, 4).isoformat()}\nmarketCap: {info['marketCap']}"


def test_tradingagents_1300_live_snapshot():
    info = {"marketCap": 1_200_000_000_000}
    assert "2026-09-04" in get_fundamentals_like_the_issue(info)  # nothing stops it

    guarded = pit.guard(get_fundamentals_like_the_issue, scan_text=True)
    snapshot = pit.guard(lambda: yfa.snapshot(info, retrieved_at="2026-09-04T14:00Z"))
    with pit.asof("2024-05-10 16:00", tz="America/New_York"):
        with pytest.raises(pit.LookaheadError, match="2026-09-04"):
            guarded(info)
        with pytest.raises(pit.LookaheadError):
            snapshot()


# --- TradingAgents #1518 -------------------------------------------------------------------


def nike_day():
    # Times from the issue: the earnings 8-K accepted 16:15 New York (20:15 UTC) on
    # 2026-10-01; CNBC at 20:07 UTC and Yahoo Finance at 20:19 UTC.
    news = [
        {"title": "midday preview", "available_at": "2026-10-01T17:00:00Z"},
        {"title": "Nike shares sink in extended trading", "available_at": "2026-10-01T20:07:00Z"},
        {"title": "Nike reports sales below estimates", "available_at": "2026-10-01T20:19:00Z"},
    ]
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320187-26-000099"],
                "filingDate": ["2026-10-01"],
                "reportDate": ["2026-10-01"],
                "acceptanceDateTime": ["2026-10-01T20:15:00.000Z"],
                "form": ["8-K"],
                "items": ["2.02,9.01"],
                "primaryDocument": ["nke-8k.htm"],
            }
        }
    }
    return news, submissions


def in_window_like_the_issue(pub, day):
    start = dt.datetime.combine(day, dt.time(0), tzinfo=dt.timezone.utc)
    return start <= pub < start + dt.timedelta(days=1)


def test_tradingagents_1518_after_close_news_and_filings():
    news, submissions = nike_day()
    day = dt.date(2026, 10, 1)
    admitted = [n for n in news if in_window_like_the_issue(pit.to_utc(n["available_at"]), day)]
    assert len(admitted) == 3  # both after-close stories leak
    filings = edgar.filings(submissions)
    assert (pd.to_datetime(filings["filing_date"]).dt.date <= day).all()  # date compare admits it

    with pit.asof("2026-10-01 16:00", tz="America/New_York"):
        assert [n["title"] for n in pit.filter(news)] == ["midday preview"]
        assert pit.filter(filings).empty
        with pytest.raises(pit.LookaheadError):
            pit.check(news)


# --- TradingAgents #1679 -------------------------------------------------------------------


def test_tradingagents_1679_short_history_sma():
    pytest.importorskip("fin_primitives")
    rng = np.random.default_rng(1679)
    close = 150 * np.exp(np.cumsum(rng.normal(0, 0.015, 45)))
    df = pd.DataFrame({"close": close})
    # min_periods=1 (stockstats' choice) makes both "SMAs" the mean of the same 45 bars
    sma50 = df["close"].rolling(50, min_periods=1).mean().iloc[-1]
    sma200 = df["close"].rolling(200, min_periods=1).mean().iloc[-1]
    assert sma50 == sma200
    out50 = df.pit.indicator("sma", 50)
    out200 = df.pit.indicator("sma", 200)
    assert np.isnan(out50["SMA_50"].iloc[-1]) and np.isnan(out200["SMA_200"].iloc[-1])
    assert out200[["bars_available", "bars_required"]].iloc[-1].tolist() == [45, 200]


def test_tradingagents_1679_with_stockstats():
    stockstats = pytest.importorskip("stockstats")
    rng = np.random.default_rng(1679)
    c = 150 * np.exp(np.cumsum(rng.normal(0, 0.015, 45)))
    frame = stockstats.wrap(pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}))
    assert frame["close_50_sma"].iloc[-1] == pytest.approx(frame["close_200_sma"].iloc[-1])


# --- FinRL #1463 ---------------------------------------------------------------------------


def test_finrl_1463_backfilled_fundamentals():
    # The issue's minimal reproduction.
    dates = pd.date_range("2020-01-01", "2020-07-10")
    ratios = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-06-30"] * 2),
            "tic": ["AAA", "AAA", "BBB", "BBB"],
            "EPS": [1.0, 2.0, 10.0, 20.0],
        }
    )
    grid = pd.DataFrame(list(itertools.product(dates, ["AAA", "BBB"])), columns=["date", "tic"])
    full = grid.merge(ratios, how="left", on=["date", "tic"]).sort_values(["tic", "date"]).bfill(axis="rows")
    pick = lambda frame, d: frame[(frame.tic == "AAA") & (frame.date == d)]["EPS"].iloc[0]  # noqa: E731
    assert pick(full, "2020-01-02") == 1.0  # Q1 before Q1 ended
    assert pick(full, "2020-07-05") == 10.0  # BBB's number

    # fin-pit: key each quarter on its report date (Compustat rdq; here synthetic dates a
    # month after the period end), available at the end of that day in New York, and join
    # on availability per ticker.
    ratios["rdq"] = pd.to_datetime(["2020-04-28", "2020-07-28", "2020-04-30", "2020-07-30"])
    ratios["available_at"] = [
        pit.to_utc(dt.datetime.combine(d.date(), dt.time(23, 59, 59)), "America/New_York") for d in ratios["rdq"]
    ]
    grid["decision_time"] = [pit.to_utc(dt.datetime.combine(d.date(), dt.time(16)), "America/New_York") for d in grid["date"]]
    pit_full = pit.asof_join(grid, ratios.drop(columns=["date"]), "decision_time", by="tic")
    assert np.isnan(pick(pit_full, "2020-01-02"))
    assert np.isnan(pick(pit_full, "2020-04-01"))
    assert pick(pit_full, "2020-04-29") == 1.0  # the day after the report
    assert pick(pit_full, "2020-07-05") == 1.0  # still Q1: Q2 not reported, never BBB's value
    bbb = pit_full[(pit_full.tic == "BBB") & (pit_full.date == "2020-07-05")]["EPS"].iloc[0]
    assert bbb == 10.0
