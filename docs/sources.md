# Source adapters

Each adapter turns vendor data into rows with `event_time` (what the value is about),
`available_at` (when it became knowable), `vintage`, `source` and `confidence`. The rules are
conservative: when the true time is unknown, the stamp is never earlier than it.

## FRED / ALFRED (`fin_pit.adapters.fred`)

FRED's `series/observations` has two windows. `observation_start`/`observation_end` bound the
period the data describes; `realtime_start`/`realtime_end` bound the vintage and default to today.
A request bounded only by observation date returns today's revised values, including observations
published after the decision date (TradingAgents #1275).

| Field | Rule |
|---|---|
| `event_time` | observation `date`, 00:00 New York |
| `available_at` | vintage `realtime_start` day at `release_time` in New York; default end of that day (23:59:59.999999) |
| `vintage` | `realtime_start` (ISO date) |
| `confidence` | `rule` |

FRED dates are days, not times. The end-of-day default is never early; pass the series' release
time (`datetime.time(8, 30)` for CPI and payrolls) for same-day decisions.

```python
from fredapi import Fred
from fin_pit.adapters import fred

stamped = fred.from_fredapi(Fred(api_key=KEY).get_series_all_releases("CPIAUCSL"), "CPIAUCSL")
with pit.asof("2023-03-31 16:00", tz="America/New_York"):
    cpi = fred.asof_series(stamped)          # first prints, nothing released after the decision

src = fred.FredSource(Fred(api_key=KEY), release_times={"CPIAUCSL": datetime.time(8, 30)})
src.get("CPIAUCSL")                          # as of the active asof

fred.observation_params(decision_time)       # {"realtime_start": d, "realtime_end": d} for raw API calls
```

## SEC EDGAR (`fin_pit.adapters.edgar`)

`companyfacts` gives each XBRL fact a `filed` date with no time. The submissions API
(`https://data.sec.gov/submissions/CIK##########.json`) gives every accession number an
`acceptanceDateTime`. That value is UTC: Apple's 16:30 New York earnings 8-Ks read `20:30:..Z` in
July and `21:30:..Z` in January (checked against Apple's 2025-2026 filings on 2026-11-17; the
excerpt is in `tests/data`).

| Field | Rule |
|---|---|
| `event_time` | the fact's period `end`, 00:00 New York |
| `available_at` | acceptance time of the fact's accession (`exact`) |
| fallback | end of the `filed` day in New York (`bound`): EDGAR gives filings accepted after 17:30 the next business day's date, so acceptance is never after the filing date |
| `vintage` | accession number, so restatements are later vintages of the same key |

```python
facts = edgar.facts(companyfacts, [submissions, *extra_submission_pages], concepts={"EarningsPerShareDiluted"})
with pit.asof("2026-05-01 07:00", tz="America/New_York"):
    eps = pit.latest(facts, ["concept", "unit", "start", "end"])

filings = edgar.filings(submissions, forms={"8-K", "10-Q", "10-K"})
```

The adapters parse JSON you fetched; they make no requests. SEC asks automated clients to send a
descriptive User-Agent and stay under 10 requests per second. `submissions` takes the main
document, its `filings.recent` block, or the extra `CIK##########-submissions-###.json` pages
(older filings), alone or in a list. Malformed documents raise `ValueError`/`TypeError`
(fuzz-tested in `tests/test_properties.py`).

## yfinance (`fin_pit.adapters.yfinance`)

| Data | `available_at` | Confidence |
|---|---|---|
| daily bar for session D | D's close (16:00 `America/New_York` by default) + `lag` (15 minutes by default) | rule |
| with `calendar="XNYS"` (needs `exchange_calendars`) | the session's real close, so 13:00 early closes are right | rule |
| weekly / monthly bar | close of the last day it covers; an unfinished bar is in the future | rule |
| intraday bar at t, interval i | t + i + `lag` | rule |
| `session="24/7"` daily bar (crypto) | next 00:00 UTC + `lag` | rule |
| `Ticker.info`, `fast_info`, other snapshots | retrieval time (`snapshot()`) | live |

The 15-minute lag is a conservative allowance for the closing print and vendor delay, not a
measured figure; set `lag=` to your own.

```python
bars = yfinance.stamp_bars(yf.download("NVDA", period="5y", multi_level_index=False))
bars = yfinance.history("NVDA", period="5y", calendar="XNYS")   # fetch + stamp + cut to asof
info = yfinance.snapshot("NVDA")                                 # refused by any earlier asof
```

Quarterly statement tables from yfinance carry only the period end, not the publication time, so
there is no adapter for them; use the EDGAR adapter for filing-time fundamentals.
