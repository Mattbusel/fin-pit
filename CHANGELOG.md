# Changelog

## 0.1.1 (2026-11-17)

Fixes found in an independent review of 0.1.0. 0.1.0 was published only as a GitHub release, not
on PyPI; do not use it.

- **Indicators only return causal values.** `df.pit.indicator("dpo")` used pandas-ta's centred DPO,
  which reads the SMA about `length / 2` bars ahead; it is now computed with `centered=False`, and
  `centered=True` raises `LookaheadError`. `ichimoku` returned the chikou span `ICS_26`, the close
  26 bars ahead; it is dropped. Every indicator is also checked at run time by computing it on a
  probe series and on the same series cut short, and any output column whose earlier values change
  is dropped (`pit.noncausal_columns`). A test runs the same check on all 64 fin-primitives
  indicators.
- **Non-US and crypto daily bars are no longer stamped early.** `stamp_bars` converted an aware
  index to New York, which moved a Shanghai, Tokyo or London session onto the previous day and
  stamped it 10 to 19 hours before its close. The session day now comes from the index's own zone,
  and a zone other than New York needs `calendar=` (for example `"XSHG"`) or `close_time=`,
  otherwise `ValueError`. A UTC daily index needs `session="24/7"` (crypto), `tz=` or `calendar=`.
- **Date arguments.** `guard` now reads `YYYYMMDD` ints and strings, `YYYY/MM/DD`, epoch
  milliseconds, microseconds and nanoseconds, and "now"/"today"/"latest". A millisecond epoch no
  longer crashes on Windows or passes unchecked on Linux. Naive values are read in the zone the
  `asof` was given in (was UTC), so `end_date="2024-05-10 18:00"` is refused by a 16:00 New York
  decision. A value that cannot be read warns `UnparsedDateWarning` (`unparsed="raise"` refuses it).
  `date_tz` now defaults to the `asof` zone.
- **Fail-open paths closed.** `@asof` on a generator or async generator keeps the decision time for
  every step. A guarded generator is checked item by item. Records nested in dicts, lists and
  tuples are checked. A guarded call with no active `asof` (a worker thread that did not copy the
  context) warns `UnguardedCallWarning` once; `pit.bind(fn)` carries the decision time into a
  thread pool. `scan_text` reads offset-less times in the `asof` zone, compares bare dates with the
  decision's day in that zone, and also matches `YYYY/MM/DD`, `May 11, 2024` and `11 May 2024`.
- **Indicators inside `asof` on plain OHLCV frames.** The docs said the frame was cut first; for a
  frame without `available_at` nothing was cut. Rows labelled at or after the decision time are now
  dropped with an `UnstampedFrameWarning`, and `unstamped="raise"` refuses the frame.
- **LangChain tools.** `guard_tool` forwards the run's config and, when the original needs one, a
  full `ToolCall`: tools with an `InjectedToolCallId` argument work (they failed on every call), and
  `response_format="content_and_artifact"` tools keep their artifact, which is now checked or
  filtered too. LangGraph `Command` results pass through.
- **Smaller fixes.** `required_bars` sizes its probe from the parameters, so `sma(5000)` works. EDGAR
  `acceptanceDateTime` values that are numeric or empty fall back to the `bound` rule instead of
  reading as 1970 or raising. `pd.NA` and `"NaT"` strings count as missing. `latest` on pandas no
  longer picks a row with no `available_at` as the newest vintage (it now matches Polars).
- **Docs.** The canonical vectors match TA-Lib 0.8.3 on the 41 indicators TA-Lib implements (it has
  no HMA or VWMA), not "all 43"; fin-primitives 3.7.0 has 64 indicators, not "about 70". Error
  messages point to `fin_pit.warmup.indicators()`.
- `fin-primitives>=3.7,<4`; a test asserts the private `fin_primitives.pandas._inputs` still
  exists, and if it disappears coverage counts every OHLCV column (stricter, never looser).

## 0.1.0 (2026-11-17)

First release.

- `asof(decision_time)`: context manager and decorator (sync and async) holding the decision time
  in a ContextVar; nesting may only move the clock back.
- `filter`, `check`, `latest` over pandas DataFrames/Series, Polars DataFrames/LazyFrames, lists of
  `PITRecord` or dicts, and single records; `LookaheadError`, `UnstampedError`.
- `PITRecord(event_time, available_at, vintage, source, confidence)` with confidence levels
  `exact`, `rule`, `bound`, `live`.
- `asof_join`: per-key join on availability (fix for back-filled fundamentals).
- `guard` for any callable (date arguments, stamped outputs, opt-in text scan) and `guard_tool` /
  `guard_tools` for LangChain and LangGraph tools; the raised error is also a `ToolException`.
- `df.pit.indicator(name, ...)`: fin-primitives indicators with strict warmup and
  `bars_available` / `bars_required`; `required_bars`, `coverage`.
- Adapters: FRED/ALFRED vintages (fredapi), SEC EDGAR acceptance times (submissions +
  companyfacts), yfinance bars (close + lag, exchange_calendars closes, 24/7 sessions) and live
  snapshots.
- Conformance spec 1.0: 43 indicators, 8 fixtures, 568 cases with named variants;
  `fin_pit.conformance.cases`, `parametrize`, `results`. Canonical vectors match TA-Lib 0.8.3 on
  every indicator it implements; results for pandas-ta-classic, fin-primitives and stockstats in
  docs/conformance-results.md.
- Tests reproducing TradingAgents #1275, #1300, #1518, #1679 and FinRL #1463, with notebooks.
