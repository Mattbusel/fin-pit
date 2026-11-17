# Reproduced look-ahead reports

Each report below has a test in `tests/test_issues.py` and a notebook in `examples/`. Each test
runs the code shape the report describes and shows the leak, then shows fin-pit stopping it.
Where a report gives real dates or times, the fixtures use them; values the report does not give
are synthetic and marked as such.

| Report | Leak | fin-pit |
|---|---|---|
| [TradingAgents #1275](https://github.com/TauricResearch/TradingAgents/issues/1275) | FRED requests bounded by observation date only, so a run dated 2023-03-31 gets March CPI (released 2023-04-12) at revised values | `adapters.fred.from_fredapi` + `asof_series`: only vintages released by the decision time, first prints |
| [TradingAgents #1300](https://github.com/TauricResearch/TradingAgents/issues/1300) | a fundamentals snapshot stamped with `datetime.now()` inside a 2024 analysis | `adapters.yfinance.snapshot` stamps `live`; `guard(..., scan_text=True)` refuses text dated after the decision |
| [TradingAgents #1518](https://github.com/TauricResearch/TradingAgents/issues/1518) | news and EDGAR filings released after the 16:00 close admitted to a decision executed at that close (calendar-day windows) | `asof("... 16:00", tz="America/New_York")` with acceptance times from `adapters.edgar` and article times |
| [TradingAgents #1679](https://github.com/TauricResearch/TradingAgents/issues/1679) | 50 and 200 SMAs computed from about 45 sessions come out equal | `df.pit.indicator("sma", 200)`: NaN, `bars_available=45`, `bars_required=200` |
| [FinRL #1463](https://github.com/AI4Finance-Foundation/FinRL/issues/1463) | quarterly ratios keyed on the fiscal period end and back-filled across the whole frame, so values appear before the quarter ends and bleed between tickers | `asof_join(grid, ratios, "decision_time", by="tic")` keyed on the report date |

These tests use small fixtures and run in CI without network access.
