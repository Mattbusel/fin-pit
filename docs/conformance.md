# Using the conformance vectors

The vectors are defined in [SPEC.md](../SPEC.md): 43 indicators, 8 fixture series, every
combination of each indicator's named variants, 568 cases. Measured results for TA-Lib,
pandas-ta-classic, fin-primitives and stockstats are in [conformance-results.md](conformance-results.md).

## In a test suite

```python
from fin_pit import conformance

@conformance.parametrize("rsi", smoothing="wilder")
def test_rsi_matches_spec(pit_case):
    out = mylib.rsi(pit_case.inputs["close"], pit_case.params["length"])
    pit_case.check(out)

@conformance.parametrize("ema", seed="adjusted")
def test_ema_matches_spec(pit_case):
    pit_case.check(mylib.ema(pit_case.inputs["close"], 20), warmup="partial")

@conformance.parametrize("bbands", ddof=0)
def test_bbands(pit_case):
    df = mylib.bbands(pit_case.inputs["close"], 20, 2.0)
    pit_case.check(df, outputs={"lower": "BBL", "middle": "BBM", "upper": "BBU"})
```

`parametrize(indicator, argname="pit_case", **variant)` is `pytest.mark.parametrize` over the
matching cases with readable ids such as `rsi/smoothing=wilder/flat`. Axes you do not name take
their canonical option.

## `Case`

| Attribute | |
|---|---|
| `id`, `indicator`, `fixture` | names |
| `params` | the parameters the vector was computed with |
| `variant` | the full variant, every axis |
| `inputs` | `{"open", "high", "low", "close", "volume"}` lists for the fixture |
| `args()` | the indicator's inputs in catalogue order |
| `expected` | `{output: list}` with NaN in warmup positions |
| `first_valid` | `{output: index}` |
| `zero_div` | `{output: set of indices}` whose canonical value came from a zero denominator |

`case.check(actual, warmup="nan", zero_division="talib", rtol=1e-9, atol=1e-9, outputs=None)`
raises `ConformanceError` listing the first mismatches; `case.mismatches(...)` returns the list.
`actual` may be one array-like, a tuple/list of arrays in catalogue output order, a dict keyed by
output name, or a DataFrame (columns in order, or mapped with `outputs=`).

## Whole-indicator runs

```python
report = conformance.results(lambda case: mylib.adx(*case.args(), 14), "adx")
failed = {cid: problems for cid, problems in report.items() if problems}
```

## Declaring a variant

Pick the options that describe your implementation; that set is your declaration. For example
stockstats declares `rsi: smoothing=wilder_adjusted, warmup=partial`, and pandas-ta-classic
declares `macd: seed=independent, align=each` and `rsi: smoothing=wilder, zero_division=0.0`.
If no option describes your behaviour, open an issue with the formula: new variants go into a
minor spec version.
