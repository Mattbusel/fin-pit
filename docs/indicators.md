# Strict-warmup indicators

```python
import fin_pit  # registers df.pit

out = df.pit.indicator("sma", 200)
out = df.pit.indicator("macd", 12, 26, 9)
out = df.pit.indicator("rsi", 14, settle=50)
out = df.pit.indicator("atr", 14, close="adj_close")
```

`indicator(name, *args, settle=0, **params)` calls fin-primitives' `df.fp.<name>(*args, **params)`
(pandas-ta's names, defaults and formulas, computed in Rust; about 70 indicators, see
`df.fp.indicators()`), then:

1. inside `pit.asof`, drops rows of the frame that were not available at the decision time;
2. computes `bars_available`: consecutive bars, ending at each row, whose inputs (only the columns
   that indicator reads) are all present. A missing value resets the count;
3. computes `bars_required`: how many bars the indicator needs before its first value, plus
   `settle`;
4. sets every output to NaN where `bars_available < bars_required`.

The result is a DataFrame: the indicator columns (fin-primitives / pandas-ta names), then
`bars_available` and `bars_required`.

## How `bars_required` is found

`pit.required_bars(name, *args, **params)` runs the indicator on a clean, deterministic 512-bar
OHLCV random walk and reads off the first row where every output column has a value. It is cached
per parameter set. For the indicators that are also in the conformance set it equals the spec's
warmup (`tests/test_warmup.py` checks sma, ema, rsi, atr, wma, dema, tema, cci, willr, mfi, adx).

## `settle`

Recursive indicators (EMA, RSI, ATR, ADX) emit their first value as soon as the seed window is
full, but that value still depends on where the series started. `settle=k` asks for `k` more bars
before trusting the output. Choose `k` from your own tolerance; fin-pit does not pick one for you.

## Why NaN and not a partial value

stockstats computes rolling windows with `min_periods=1`, by design, so a 200-bar SMA on 45 bars
is the 45-bar mean (TradingAgents #1679 shows the 50 and 200 SMAs coming out equal). In the
conformance spec that is the declared variant `warmup=partial`, not an error. In a backtest it is
a different indicator from the one the strategy names, and strict warmup makes the shortfall
visible: `bars_available=45, bars_required=200`.
