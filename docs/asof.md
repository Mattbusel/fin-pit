# The point-in-time firewall

## Times

Every time in fin-pit is an aware UTC `datetime`. Input can be an aware `datetime`, a pandas
`Timestamp`, a `numpy.datetime64` (read as UTC), an ISO 8601 string with an offset, or a POSIX
timestamp. A naive time or a bare date needs `tz=`:

```python
pit.to_utc("2024-05-10 16:00", "America/New_York")   # 20:00 UTC
pit.to_utc(datetime.date(2024, 5, 10), "America/New_York")   # 00:00 New York, the earliest moment of the day
pit.to_utc("2024-05-10 16:00")   # ValueError: no time zone
```

Refusing naive times is deliberate: a naive "2024-05-10" is midnight UTC to one vendor and the
New York session to another, and that gap is where after-close news leaks in.

## `asof`

```python
with pit.asof("2024-05-10 16:00", tz="America/New_York"):
    ...

@pit.asof("2024-05-10T20:00Z", mode="filter")
async def run_agent(): ...
```

| Option | Default | Meaning |
|---|---|---|
| `mode` | `"raise"` | what a guarded read does with late rows: raise `LookaheadError`, or `"filter"` them out |
| `unstamped` | `"allow"` | values with no `available_at`: pass through, or `"raise"` `UnstampedError` |
| `column` | `"available_at"` | the column or index level that holds availability times |

The decision time lives in a `contextvars.ContextVar`: it follows `await`, does not leak into
other threads or tasks, and `contextvars.copy_context().run(...)` or `pit.bind(fn)` carries it
into a worker (`executor.submit(pit.bind(fetch), "NVDA")`). Nested blocks may only move the clock
back; a nested `asof` later than its parent raises.

As a decorator, `asof` works on plain, async, generator and async generator functions; a
generator body sees the decision time on every step, not only until its first `yield`.

`asof` also remembers the zone the decision was given in (`tz=`, or the offset of an aware
time). Guarded calls read naive date arguments, and `scan_text` reads naive timestamps, in that
zone.

`pit.decision_time()` returns the active time (or `None`); `pit.current()` returns the whole state.

## `filter`, `check`, `latest`

| Call | Late rows | No decision time |
|---|---|---|
| `pit.filter(obj, when=None)` | dropped | `obj` unchanged |
| `pit.check(obj, when=None)` | `LookaheadError` | `obj` unchanged |
| `pit.latest(obj, on, when=None)` | dropped, then the newest vintage per `on` key | newest vintage overall |

`when` is optional inside `asof`; outside it, it sets the decision time for that call. An explicit
`when` later than the active `asof` raises.

Supported objects:

- pandas DataFrame with an `available_at` column or index level (tz-aware; naive raises; strings
  are parsed and must carry an offset);
- pandas Series with an `available_at` index level;
- Polars DataFrame (`check` and `filter`) and LazyFrame (`filter` only, so nothing is collected);
- lists and tuples of `PITRecord`, dicts with an `available_at` key, or objects with an
  `available_at` attribute;
- a single record (a late one raises even under `filter`, since it cannot be dropped);
- dicts, lists and tuples that nest any of the above (`{"rows": [...], "meta": ...}`,
  `(content, artifact)`): each nested container is checked or filtered in place.

A missing `available_at` (`None`, `NaN`, `NaT`, `pd.NA`, or the strings `""`/`"NaT"`) means
"unknown" and is treated as not available. `latest` never picks such a row over a real vintage.

`LookaheadError` carries `decision_time`, `available_at` (the earliest late time) and `count`.

## `PITRecord`

```python
pit.PITRecord(event_time, available_at, vintage=None, source="", confidence="rule", value=None, key=None)
```

`confidence` is one of `exact` (the source publishes the moment), `rule` (derived by a documented
rule), `bound` (a conservative upper bound) and `live` (the source only serves the current value).

## `asof_join`

```python
pit.asof_join(decisions, facts, "decision_time", by="tic")
```

For each left row, the right row with the greatest `available_at` at or before the left row's
decision time, per `by` key. Nothing is back-filled and keys never bleed into each other. This is
the fix for fundamentals joined on fiscal period ends (FinRL #1463).

## `guard`

```python
@pit.guard(mode="filter")
def get_news(ticker, curr_date): ...
```

`pit.guard(fn, **options)` wraps any callable (sync, async, generator or async generator).
Inside `asof`:

1. arguments named in `date_args` (`as_of`, `curr_date`, `current_date`, `trade_date`, `end`,
   `end_date`, `end_time`, `date`, `to`, `to_date`, `until`, ...) that read as a time later than
   the decision time raise before the call. Understood: ISO dates and times, `YYYYMMDD` ints and
   strings (tushare, akshare), `YYYY/MM/DD`, epoch seconds, milliseconds (ccxt `until`),
   microseconds and nanoseconds, `date`/`datetime`/`Timestamp`/`datetime64`, and "now", "today",
   "latest" (the wall clock). Naive values are read in `date_tz`, by default the zone the `asof`
   was given in, so `end_date="2024-05-10 18:00"` is refused by a 16:00 New York decision. A bare
   date means "up to and including that day" and passes on the decision day itself; the rows that
   come back are still checked. A value that cannot be read warns `UnparsedDateWarning`
   (`unparsed="raise"` refuses it, `"ignore"` skips it);
2. the result goes through `filter` or `check` (`mode`), including nested containers; a generator
   result is checked item by item as it is iterated (in `filter` mode a late record is skipped);
3. with `scan_text=True`, a string result that mentions a later date raises. Matched:
   `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY.MM.DD` (with an optional time and offset), `May 11, 2024`
   and `11 May 2024`. A bare date counts only when the whole day is after the decision's day in
   the `asof` zone; a time without an offset is read in that zone. Numeric dates such as
   `05/11/2024` are ambiguous (month or day first) and are not matched.

Outside `asof` nothing can be checked. `require_asof=True` raises, `require_asof=False` passes
silently, and the default passes with a one-time `UnguardedCallWarning` per wrapped function.

## Polars

```python
df.pit.asof(t); df.pit.check(t); df.pit.latest("event_time", t)
lazy.pit.asof(t)
```
