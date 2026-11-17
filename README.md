# fin-pit

Point-in-time guards for backtests and trading agents.

A backtest dated March 31 that reads March CPI, an agent that reads a 16:15 earnings 8-K before
trading at the 16:00 close, a "200-day SMA" computed from 45 days: each is a look-ahead bug, and
each agent-finance repo fixes its own copy by hand. fin-pit makes "when could this have been
known" a value you can check:

- every record carries `available_at`, the first moment it was knowable;
- code runs under `pit.asof(decision_time)`, and guarded reads refuse (or drop) anything later;
- indicators emit NaN until they have the bars they need, with the coverage beside the value;
- adapters stamp FRED/ALFRED vintages, SEC EDGAR acceptance times and yfinance bars.

```bash
pip install fin-pit
```

## The firewall

```python
import fin_pit as pit

with pit.asof("2026-10-01 16:00", tz="America/New_York"):
    news = pit.filter(news_rows)        # drops items published after the close
    pit.check(filings)                  # raises LookaheadError if any filing is later
    fundamentals = pit.latest(xbrl, ["concept", "unit", "start", "end"])  # newest vintage known then
```

Naive timestamps are refused. `pit.asof` is a context manager and a decorator, follows
`await`, and cannot be nested later than its parent.

## Agent tools (LangChain, LangGraph)

```python
tools = pit.guard_tools([get_news, get_fundamentals, get_prices], mode="filter")
agent = create_react_agent(model, tools)          # same names, descriptions and schemas

with pit.asof(trade_time):
    agent.invoke({"messages": [...]})
```

A guarded tool refuses date arguments (`curr_date`, `end_date`, ...) later than the decision
time before the vendor is called, then filters or checks what comes back. With
`scan_text=True`, string output that mentions a later date (a report "retrieved on" today)
raises.

## Strict-warmup indicators

```python
df.pit.indicator("sma", 200)
#      SMA_200  bars_available  bars_required
# 44       NaN              45            200
```

Computed by [fin-primitives](https://github.com/Mattbusel/fin-primitives) (pandas-ta's formulas
in Rust, about 70 indicators); fin-pit masks every row that has fewer bars behind it than the
indicator needs. A gap in the inputs resets the count. Inside `asof` the frame is cut first.

## Sources

| Source | `available_at` rule | Confidence |
|---|---|---|
| FRED / ALFRED (`adapters.fred`) | end of the vintage's `realtime_start` day in New York, or a release time you give | rule |
| SEC EDGAR (`adapters.edgar`) | the filing's `acceptanceDateTime` (UTC) | exact |
| SEC EDGAR, accession not in your submissions data | end of the `filed` day in New York | bound |
| yfinance bars (`adapters.yfinance`) | session close + lag (15 min default), real closes with `exchange_calendars` | rule |
| yfinance `Ticker.info` snapshots | retrieval time | live |

## Conformance vectors

43 indicators x 8 fixture series x every named variant = 568 golden cases (`fin_pit.conformance`,
spec in [SPEC.md](https://github.com/Mattbusel/fin-pit/blob/main/SPEC.md)). Libraries legitimately differ (Wilder vs EMA-smoothed RSI, SMA-seeded
vs adjusted EMA, NaN warmup vs computing from the bars available), so a library declares its
variant and is checked against it:

```python
from fin_pit import conformance

@conformance.parametrize("rsi", smoothing="wilder")
def test_rsi(pit_case):
    pit_case.check(my_rsi(pit_case.inputs["close"], 14))
```

The canonical vectors match TA-Lib 0.8.3 on all 43 indicators; pandas-ta-classic 0.9.0,
fin-primitives 3.7.0 and stockstats 0.6.14 results per indicator are in
[docs/conformance-results.md](https://github.com/Mattbusel/fin-pit/blob/main/docs/conformance-results.md).

## Docs

- [docs/asof.md](https://github.com/Mattbusel/fin-pit/blob/main/docs/asof.md): the firewall, modes, nesting, async, Polars
- [docs/agents.md](https://github.com/Mattbusel/fin-pit/blob/main/docs/agents.md): guarding LangChain / LangGraph tools
- [docs/indicators.md](https://github.com/Mattbusel/fin-pit/blob/main/docs/indicators.md): strict warmup and coverage
- [docs/sources.md](https://github.com/Mattbusel/fin-pit/blob/main/docs/sources.md): FRED/ALFRED, EDGAR and yfinance rules
- [docs/conformance.md](https://github.com/Mattbusel/fin-pit/blob/main/docs/conformance.md): using the vectors in your test suite
- [docs/issues.md](https://github.com/Mattbusel/fin-pit/blob/main/docs/issues.md): the five look-ahead reports these tests reproduce
- [examples/](https://github.com/Mattbusel/fin-pit/tree/main/examples): one notebook per reproduced issue

MIT licensed.
