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
| nested output | records inside dicts and lists (`{"rows": [...]}`), generators (item by item) and a `ToolMessage` artifact (`response_format="content_and_artifact"`) are checked the same way |
| text output | with `scan_text=True`, a later date in the returned string raises |
| unstamped output | passes through, or raises with `unstamped="raise"` |
| unreadable date argument | warns `UnparsedDateWarning` (`unparsed="raise"` refuses it) |

The wrapper is a LangChain `StructuredTool` with the original `name`, `description`,
`args_schema`, `response_format`, `return_direct`, `handle_tool_error` and `tags`, so the model
sees the same schema. It calls the original tool's `invoke`/`ainvoke` with the run's config, so
the original validation, callbacks and `RunnableConfig` arguments work. When the original needs a
full `ToolCall` (an `InjectedToolCallId` argument, as in LangGraph handoffs, or
`response_format="content_and_artifact"`), the wrapper rebuilds one with the same id. A LangGraph
`Command` returned by a tool passes through unchecked: it carries state updates, not data rows.

## Threads

The decision time is a context variable. LangGraph's `ToolNode` carries it into its own workers
(checked for both the sync and async paths), but a plain `ThreadPoolExecutor.submit` does not.
A guarded call that finds no decision time passes through with a one-time
`UnguardedCallWarning`; submit `pit.bind(fn)` instead so the check runs, or pass
`require_asof=True` to make the missing decision time an error.

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
