import asyncio

import pandas as pd
import pytest

import fin_pit as pit

lc = pytest.importorskip("langchain_core.tools")

ROWS = pd.DataFrame(
    {
        "headline": ["before the close", "after the close"],
        "available_at": pd.to_datetime(["2026-10-01T19:30Z", "2026-10-01T20:07Z"], utc=True),
    }
)


@lc.tool
def get_news(ticker: str, curr_date: str) -> str:
    """Headlines for a ticker up to curr_date."""
    return ROWS


@lc.tool
def get_text(ticker: str, curr_date: str) -> str:
    """A text report."""
    return f"# Data retrieved on: 2026-09-04\n{ticker}"


def test_guard_tool_keeps_identity():
    g = pit.guard_tool(get_news)
    assert isinstance(g, lc.BaseTool)
    assert g.name == "get_news" and g.description == get_news.description
    assert g.args == get_news.args


def test_guard_tool_raises_and_filters():
    g = pit.guard_tool(get_news)
    f = pit.guard_tool(get_news, mode="filter")
    with pit.asof("2026-10-01 16:00", tz="America/New_York"):
        with pytest.raises(pit.LookaheadError):
            g.invoke({"ticker": "NKE", "curr_date": "2026-10-01"})
        with pytest.raises(pit.LookaheadError):
            f.invoke({"ticker": "NKE", "curr_date": "2026-10-02"})
        out = f.invoke({"ticker": "NKE", "curr_date": "2026-10-01"})
    assert out["headline"].tolist() == ["before the close"]
    # outside asof the tool behaves exactly as before
    assert len(g.invoke({"ticker": "NKE", "curr_date": "2026-10-01"})) == 2


def test_guard_tool_async_and_text_scan():
    g = pit.guard_tool(get_text, scan_text=True)

    async def run():
        with pit.asof("2024-05-10T20:00Z"):
            return await g.ainvoke({"ticker": "BTC-USD", "curr_date": "2024-05-10"})

    with pytest.raises(pit.LookaheadError):
        asyncio.run(run())


def test_guard_tool_error_handling_flag():
    g = pit.guard_tool(get_news)
    g.handle_tool_error = True
    with pit.asof("2026-10-01 16:00", tz="America/New_York"):
        msg = g.invoke({"ticker": "NKE", "curr_date": "2026-10-01"})
    assert "not available" in msg


def test_guard_tools_and_plain_callables():
    tools = pit.guard_tools([get_news, lambda curr_date: [{"available_at": "2030-01-01T00:00Z"}]])
    assert tools[0].name == "get_news"
    with pit.asof("2024-01-01T00:00Z"):
        with pytest.raises(pit.LookaheadError):
            tools[1](curr_date="2023-12-31")
    with pytest.raises(TypeError):
        pit.guard_tool(42)
