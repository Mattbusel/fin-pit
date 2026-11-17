# Strict-warmup indicators

```python
import fin_pit  # registers df.pit

out = df.pit.indicator("sma", 200)
out = df.pit.indicator("macd", 12, 26, 9)
out = df.pit.indicator("rsi", 14, settle=50)
out = df.pit.indicator("atr", 14, close="adj_close")
```

`indicator(name, *args, settle=0, **params)` calls fin-primitives' `df.fp.<name>(*args, **params)`
(pandas-ta's names, defaults and formulas, computed in Rust; 64 indicators in fin-primitives
3.7.0, listed by `fin_pit.warmup.indicators()`), then:

1. inside `pit.asof`, cuts the frame to the decision time. A frame with an `available_at` column
   or index level loses the rows not available then. A plain OHLCV frame has no availability
   times, and a bar's label is when it opened, not when it closed: rows labelled at or after the
   decision time are dropped and an `UnstampedFrameWarning` says the last bar may still be open
   (`unstamped="raise"` refuses the frame instead). Stamp bars with
   `fin_pit.adapters.yfinance.stamp_bars` to cut exactly;
2. computes `bars_available`: consecutive bars, ending at each row, whose inputs (only the columns
   that indicator reads) are all present. A missing value resets the count;
3. computes `bars_required`: how many bars the indicator needs before its first value, plus
   `settle`;
4. sets every output to NaN where `bars_available < bars_required`.

The result is a DataFrame: the indicator's causal columns (fin-primitives / pandas-ta names),
then `bars_available` and `bars_required`.

## Only causal outputs

Two pandas-ta conventions read later bars, which warmup masking cannot fix because it only trims
the start of the series:

- `dpo` is centred by default (it compares the close with the SMA about `length / 2` bars
  ahead). fin-pit computes it with `centered=False`; passing `centered=True` raises
  `LookaheadError`.
- `ichimoku` returns the chikou span `ICS_26`, the close 26 bars ahead. fin-pit drops it.

Beyond those two, every indicator is checked when first used: it is computed on the probe series
and on the same series cut short, and any output column whose earlier values change is dropped
(`pit.noncausal_columns(name, ...)` lists them; an indicator with no causal column raises).
`tests/test_review_fixes.py` runs the same truncate-and-compare check on `df.pit.indicator` for
every fin-primitives indicator, so a new non-causal one fails CI.

## How `bars_required` is found

`pit.required_bars(name, *args, **params)` runs the indicator on a clean, deterministic OHLCV
random walk (512 bars, or at least four times the largest numeric parameter, so `sma(5000)` on
minute bars works) and reads off the first row where every output column has a value. It is cached
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
