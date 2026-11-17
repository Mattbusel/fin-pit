import datetime as dt
import json
import os

import pandas as pd
import pytest

import fin_pit as pit
from fin_pit.adapters import edgar, fred, yfinance as yfa

UTC = dt.timezone.utc
DATA = os.path.join(os.path.dirname(__file__), "data")


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


# --- FRED / ALFRED ----------------------------------------------------------------------------


def fredapi_frame():
    """Shaped like fredapi.Fred.get_series_all_releases(): date, realtime_start, value.
    Release dates for Feb and Mar 2023 CPI are the ones in TradingAgents #1275; the values
    are synthetic."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-02-01", "2023-03-01", "2023-03-01"]),
            "realtime_start": pd.to_datetime(["2023-02-14", "2023-03-14", "2024-02-13", "2023-04-12", "2024-02-13"]),
            "value": [100.0, 100.4, 100.5, 100.5, 100.6],
        }
    )


def test_fred_available_at_rules():
    assert fred.available_at("2023-04-12") == dt.datetime(2023, 4, 13, 3, 59, 59, 999999, tzinfo=UTC)
    assert fred.available_at("2023-04-12", dt.time(8, 30)) == dt.datetime(2023, 4, 12, 12, 30, tzinfo=UTC)
    with pytest.raises(TypeError):
        fred.available_at("2023-04-12", "08:30")


def test_fred_from_fredapi_and_asof_series():
    stamped = fred.from_fredapi(fredapi_frame(), "CPIAUCSL")
    assert list(stamped.columns) == ["series_id", "event_time", "available_at", "vintage", "value", "source", "confidence"]
    with pit.asof("2023-03-31 16:00", tz="America/New_York"):
        s = fred.asof_series(stamped)
    assert s.index.strftime("%Y-%m-%d").tolist() == ["2023-01-01", "2023-02-01"]
    assert s.tolist() == [100.0, 100.4]  # the first print, not the 2024 revision
    latest = fred.asof_series(stamped)
    assert latest.tolist() == [100.0, 100.5, 100.6]
    recs = fred.records_from_fredapi(fredapi_frame(), "CPIAUCSL")
    assert recs[0].confidence == "rule" and recs[0].key == "CPIAUCSL"


def test_fred_observation_params():
    # 16:00 on 2023-04-12 is before the end-of-day default release time: previous vintage.
    p = fred.observation_params(pit.to_utc("2023-04-12 16:00", "America/New_York"))
    assert p == {"realtime_start": "2023-04-11", "realtime_end": "2023-04-11"}
    p = fred.observation_params(pit.to_utc("2023-04-12 16:00", "America/New_York"), dt.time(8, 30))
    assert p == {"realtime_start": "2023-04-12", "realtime_end": "2023-04-12"}


def test_fred_source_wraps_client():
    class FakeFred:
        calls = 0

        def get_series_all_releases(self, series_id):
            FakeFred.calls += 1
            return fredapi_frame()

    src = fred.FredSource(FakeFred(), release_times={"CPIAUCSL": dt.time(8, 30)})
    with pit.asof("2023-04-12 09:00", tz="America/New_York"):
        assert src.get("CPIAUCSL").tolist() == [100.0, 100.4, 100.5]
    assert src.get("CPIAUCSL", "2023-03-01T00:00Z").tolist() == [100.0]
    assert FakeFred.calls == 1
    with pytest.raises(TypeError):
        fred.FredSource(object())


def test_fred_bad_input():
    with pytest.raises(TypeError):
        fred.from_fredapi([1, 2], "X")
    with pytest.raises(ValueError):
        fred.from_fredapi(pd.DataFrame({"date": [], "value": []}), "X")


# --- SEC EDGAR -----------------------------------------------------------------------------


def test_edgar_acceptance_times_are_utc():
    sub = load("aapl_submissions_excerpt.json")
    acc = edgar.acceptance_times(sub)
    # Apple's earnings 8-K: released 16:30 New York, EDT in July (UTC-4), EST in January (UTC-5).
    assert acc["0000320193-26-000018"] == dt.datetime(2026, 7, 30, 20, 30, 28, tzinfo=UTC)
    assert acc["0000320193-26-000005"] == dt.datetime(2026, 1, 29, 21, 30, 33, tzinfo=UTC)
    ny = acc["0000320193-26-000018"].astimezone(pit._time.zone("America/New_York"))
    assert (ny.hour, ny.minute) == (16, 30)


def test_edgar_filings_frame():
    f = edgar.filings(load("aapl_submissions_excerpt.json"), forms={"8-K"})
    assert set(f["form"]) == {"8-K"} and (f["confidence"] == "exact").all()
    assert all("2.02" in i for i in f["items"])
    with pit.asof("2026-07-30 16:00", tz="America/New_York"):
        known = pit.filter(f)
    assert "0000320193-26-000018" not in set(known["accession"])  # accepted 16:30, after the close


def test_edgar_facts_and_vintages():
    facts = edgar.facts(load("aapl_companyfacts_excerpt.json"), load("aapl_submissions_excerpt.json"))
    assert len(facts) == 16
    # accession 0000320193-25-000057 is older than the submissions excerpt: bounded by its filed day
    old = facts[facts.accession == "0000320193-25-000057"]
    assert (old.confidence == "bound").all()
    assert old.available_at.iloc[0] == pd.Timestamp("2025-05-03T03:59:59.999999Z")
    q = facts[(facts.start == "2025-12-28") & (facts.end == "2026-03-28")]
    assert q.available_at.iloc[0] == pd.Timestamp("2026-05-01T10:01:00Z")
    keys = ["concept", "unit", "start", "end"]
    with pit.asof("2026-05-01 05:00", tz="America/New_York"):  # 09:00Z, before the 10:01Z acceptance
        before = pit.latest(facts, keys)
    with pit.asof("2026-05-01 07:00", tz="America/New_York"):
        after = pit.latest(facts, keys)
    assert ("2025-12-28", "2026-03-28") not in set(zip(before.start, before.end))
    assert ("2025-12-28", "2026-03-28") in set(zip(after.start, after.end))
    recs = edgar.fact_records(load("aapl_companyfacts_excerpt.json"), load("aapl_submissions_excerpt.json"))
    assert recs[0].source == "sec-edgar" and recs[0].key.startswith("us-gaap:EarningsPerShareDiluted[USD/shares]")


def test_edgar_without_submissions_is_bounded():
    facts = edgar.facts(load("aapl_companyfacts_excerpt.json"))
    assert (facts.confidence == "bound").all()
    exact = edgar.facts(load("aapl_companyfacts_excerpt.json"), load("aapl_submissions_excerpt.json"))
    both = facts.merge(exact, on=["accession", "start", "end"], suffixes=("_b", "_e"))
    assert (both.available_at_e <= both.available_at_b).all()  # the bound is never early


@pytest.mark.parametrize(
    "doc",
    [
        {},
        {"filings": {}},
        {"filings": {"recent": {"accessionNumber": ["a", "b"], "acceptanceDateTime": ["2024-01-01T00:00:00.000Z"]}}},
        {"filings": {"recent": {"accessionNumber": "a"}}},
        [1],
    ],
)
def test_edgar_rejects_malformed_submissions(doc):
    with pytest.raises((ValueError, TypeError)):
        edgar.acceptance_times(doc)


@pytest.mark.parametrize(
    "doc",
    [
        {},
        {"facts": {"us-gaap": []}},
        {"facts": {"us-gaap": {"X": {"units": {"USD": {}}}}}},
        {"facts": {"us-gaap": {"X": {"units": {"USD": [1]}}}}},
        {"facts": {"us-gaap": {"X": {"units": {"USD": [{"val": 1}]}}}}},
    ],
)
def test_edgar_rejects_malformed_facts(doc):
    with pytest.raises((ValueError, TypeError)):
        edgar.facts(doc)


# --- yfinance -----------------------------------------------------------------------------


def daily(n=5, tz=None):
    idx = pd.date_range("2024-05-06", periods=n, freq="B", tz=tz)
    return pd.DataFrame({"Close": range(n)}, index=idx)


def test_stamp_daily_bars():
    out = yfa.stamp_bars(daily(tz="America/New_York"))
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-05-06T20:15Z")  # 16:00 EDT + 15 min
    assert (out["confidence"] == "rule").all()
    with pit.asof("2024-05-08 16:00", tz="America/New_York"):
        assert len(pit.filter(out)) == 2  # the 05-08 bar is not final until 16:15
    out0 = yfa.stamp_bars(daily(), lag=dt.timedelta(0))
    assert out0["available_at"].iloc[0] == pd.Timestamp("2024-05-06T20:00Z")


def test_stamp_intraday_and_crypto_and_weekly():
    idx = pd.date_range("2024-05-06 09:30", periods=3, freq="5min", tz="America/New_York")
    out = yfa.stamp_bars(pd.DataFrame({"Close": [1, 2, 3]}, index=idx), "5m", lag=dt.timedelta(0))
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-05-06T13:35Z")
    c = yfa.stamp_bars(pd.DataFrame({"Close": [1]}, index=pd.DatetimeIndex(["2024-05-10"], tz="UTC")), session="24/7")
    assert c["available_at"].iloc[0] == pd.Timestamp("2024-05-11T00:15Z")
    w = yfa.stamp_bars(pd.DataFrame({"Close": [1, 2]}, index=pd.DatetimeIndex(["2024-05-06", "2024-05-13"])), "1wk")
    assert w["available_at"].iloc[0] == pd.Timestamp("2024-05-10T20:15Z")  # Friday close
    m = yfa.stamp_bars(pd.DataFrame({"Close": [1]}, index=pd.DatetimeIndex(["2024-02-01"])), "1mo")
    assert m["available_at"].iloc[0] == pd.Timestamp("2024-02-29T21:15Z")  # EST in February


def test_stamp_bars_validation():
    with pytest.raises(TypeError):
        yfa.stamp_bars([1])
    with pytest.raises(TypeError):
        yfa.stamp_bars(pd.DataFrame({"Close": [1]}))
    with pytest.raises(ValueError):
        yfa.stamp_bars(daily(), "7m")
    with pytest.raises(ValueError):
        yfa.stamp_bars(daily(), lag=dt.timedelta(minutes=-1))
    with pytest.raises(ValueError):
        yfa.stamp_bars(daily(), session="always")


def test_exchange_calendar_early_close():
    pytest.importorskip("exchange_calendars")
    idx = pd.DatetimeIndex(["2024-11-29"])  # day after Thanksgiving: NYSE closes 13:00
    out = yfa.stamp_bars(pd.DataFrame({"Close": [1]}, index=idx), calendar="XNYS", lag=dt.timedelta(0))
    assert out["available_at"].iloc[0] == pd.Timestamp("2024-11-29T18:00Z")


def test_snapshot_is_live():
    snap = yfa.snapshot({"marketCap": 1}, retrieved_at="2026-09-04T12:00Z")
    assert snap["confidence"] == "live" and snap["marketCap"] == 1
    with pit.asof("2024-05-10T20:00Z"):
        with pytest.raises(pit.LookaheadError):
            pit.check(snap)
    assert pit.check(snap, "2026-09-05T00:00Z") is snap
    with pytest.raises(TypeError):
        yfa.snapshot(42)
