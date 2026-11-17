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
other threads or tasks, and `contextvars.copy_context().run(...)` carries it into a worker.
Nested blocks may only move the clock back; a nested `asof` later than its parent raises.

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
- a single record (a late one raises even under `filter`, since it cannot be dropped).

A missing `available_at` (`None`, `NaN`, `NaT`) means "unknown" and is treated as not available.

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

`pit.guard(fn, **options)` wraps any callable (sync or async). Inside `asof`:

1. arguments named in `date_args` (`as_of`, `curr_date`, `current_date`, `trade_date`, `end`,
   `end_date`, `end_time`, `date`, `to`, `to_date`, `until`, ...) that parse as a time later than
   the decision time raise before the call; naive values are read in `date_tz` (UTC by default);
2. the result goes through `filter` or `check` (`mode`);
3. with `scan_text=True`, a string result that mentions a later date raises (a bare `YYYY-MM-DD`
   counts only when the whole day is after the decision's UTC day).

Outside `asof` the call passes through untouched, unless `require_asof=True`.

## Polars

```python
df.pit.asof(t); df.pit.check(t); df.pit.latest("event_time", t)
lazy.pit.asof(t)
```
