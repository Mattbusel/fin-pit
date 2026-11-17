# Guarding agent tools

Agent trading frameworks hand the model tools such as `get_news(ticker, curr_date)` or
`get_fundamentals(ticker)`. In a backtest each tool must answer as of the simulated trade time, and
each vendor has its own way of leaking the future: FRED serves today's vintage, a fundamentals
endpoint returns a live snapshot, a news window ends at UTC midnight instead of the close.

`pit.guard_tool` puts one check in front of all of them.

```python
import fin_pit as pit
from langgraph.prebuilt import create_react_agent

tools = pit.guard_tools([get_news, get_fundamentals, get_prices], mode="filter")
agent = create_react_agent(model, tools)

for trade_time in schedule:
    with pit.asof(trade_time):
        agent.invoke({"messages": [("user", f"Decide on NVDA for {trade_time:%Y-%m-%d}")]})
```

What the guarded tool does inside `asof`:

| Step | Effect |
|---|---|
| date arguments | `curr_date`, `end_date`, `trade_date`, ... later than the decision time raise `LookaheadError` before the vendor is called |
| stamped output | a DataFrame with `available_at`, or a list of records/dicts, is filtered (`mode="filter"`) or checked (`mode="raise"`) |
| text output | with `scan_text=True`, a later date in the returned string raises |
| unstamped output | passes through, or raises with `unstamped="raise"` |

The wrapper is a LangChain `StructuredTool` with the original `name`, `description`,
`args_schema`, `return_direct`, `handle_tool_error` and `tags`, so the model sees the same tool.
It calls the original tool's `invoke`/`ainvoke`, so the original validation and callbacks run.

## Errors the model can read

The raised error is both a `fin_pit.LookaheadError` and a LangChain `ToolException`. A tool with
`handle_tool_error=True` (or a `ToolNode` with `handle_tool_errors=True`) returns the message to
the model ("2 rows are not available at the decision time ...") instead of ending the run; code
that catches `LookaheadError` still sees it.

## Stamping your data

The guard can only check what carries a time. Make your data layer return `available_at`:

```python
from fin_pit.adapters import edgar, fred, yfinance

bars = yfinance.history("NVDA", period="5y")                 # close + 15 min
macro = fred.FredSource(fredapi.Fred(api_key=...))           # vintages
facts = edgar.facts(companyfacts_json, submissions_json)     # acceptance times
news = [{"title": a.title, "available_at": a.published_utc} for a in articles]
```

For vendors with no timestamps at all, `unstamped="raise"` turns silent leaks into errors so they
show up in testing, and `scan_text=True` catches the common "retrieved on <today>" header.

## Plain callables

`pit.guard_tool` on anything that is not a LangChain tool returns `pit.guard(fn)`, so the same
call works for CrewAI tools, OpenAI function handlers or MCP tool functions.
