"""Guard LangChain and LangGraph tools: ``guard_tool(tool)``.

``guard_tool`` returns a new tool with the same name, description and argument schema whose
calls go through ``fin_pit.guard``: date arguments later than the active ``asof`` raise
``LookaheadError`` before the vendor is called, and the result is filtered or checked on the
way out. LangGraph's ``ToolNode`` takes LangChain tools, so the guarded tool drops into a
graph unchanged.

langchain-core is only imported when a LangChain tool is passed; a plain callable is wrapped
with ``fin_pit.guard`` directly.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from .guard import DEFAULT_DATE_ARGS, LookaheadError, guard

__all__ = ["guard_tool", "guard_tools"]

_ERROR_CLASSES: dict = {}


def _tool_error_class(tool_exception: type) -> type:
    """A ``LookaheadError`` that is also LangChain's ``ToolException``, so a tool with
    ``handle_tool_error`` set hands the message back to the model, and code catching
    ``LookaheadError`` still sees it."""
    cls = _ERROR_CLASSES.get(tool_exception)
    if cls is None:
        cls = type("LookaheadToolError", (LookaheadError, tool_exception), {"__module__": __name__})
        _ERROR_CLASSES[tool_exception] = cls
    return cls


def _as_tool_error(exc: LookaheadError, cls: type) -> Exception:
    if isinstance(exc, cls):
        return exc
    err = cls(
        str(exc), decision_time=exc.decision_time, available_at=exc.available_at, count=exc.count
    )
    return err


def guard_tool(
    tool: Any,
    *,
    mode: Optional[str] = None,
    unstamped: Optional[str] = None,
    column: Optional[str] = None,
    date_args: Optional[Iterable[str]] = DEFAULT_DATE_ARGS,
    date_tz: Any = "UTC",
    scan_text: bool = False,
    require_asof: bool = False,
) -> Any:
    """Wrap a LangChain ``BaseTool`` (or any callable) in the point-in-time guard.

    Keyword arguments are those of ``fin_pit.guard``. For a ``BaseTool`` the wrapper is a
    ``StructuredTool`` that keeps ``name``, ``description``, ``args_schema``,
    ``return_direct``, ``handle_tool_error`` and ``tags``; its sync and async
    paths both call the original tool's ``invoke``/``ainvoke``, so callbacks and the
    original tool's own validation still run.
    """
    opts = dict(
        mode=mode,
        unstamped=unstamped,
        column=column,
        date_args=date_args,
        date_tz=date_tz,
        scan_text=scan_text,
        require_asof=require_asof,
    )
    try:
        from langchain_core.tools import BaseTool, StructuredTool
    except ImportError:
        BaseTool = None  # type: ignore[assignment,misc]

    if BaseTool is None or not isinstance(tool, BaseTool):
        if not callable(tool):
            raise TypeError(f"guard_tool needs a LangChain tool or a callable, got {type(tool).__name__}")
        return guard(tool, **opts)

    from langchain_core.tools import ToolException

    original = tool
    error_cls = _tool_error_class(ToolException)

    def _call(**kwargs: Any) -> Any:
        return original.invoke(kwargs)

    async def _acall(**kwargs: Any) -> Any:
        return await original.ainvoke(kwargs)

    # guard() reads date arguments by name from kwargs, which is how StructuredTool calls us.
    guarded_sync = guard(_call, **opts)
    guarded_async = guard(_acall, **opts)

    def sync_fn(**kwargs: Any) -> Any:
        try:
            return guarded_sync(**kwargs)
        except LookaheadError as exc:
            raise _as_tool_error(exc, error_cls) from exc

    async def async_fn(**kwargs: Any) -> Any:
        try:
            return await guarded_async(**kwargs)
        except LookaheadError as exc:
            raise _as_tool_error(exc, error_cls) from exc

    sync_fn.__name__ = async_fn.__name__ = str(original.name)

    # response_format is not carried over: invoke() with plain arguments returns only the
    # content, so the wrapper always answers in "content" format.
    extra = {}
    for attr in ("return_direct", "handle_tool_error", "handle_validation_error", "tags", "metadata"):
        if hasattr(original, attr):
            extra[attr] = getattr(original, attr)

    wrapped = StructuredTool.from_function(
        func=sync_fn,
        coroutine=async_fn,
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        infer_schema=False,
        **extra,
    )
    return wrapped


def guard_tools(tools: Iterable[Any], **kwargs: Any) -> List[Any]:
    """``guard_tool`` over a list (for ``create_react_agent(model, guard_tools(tools))``)."""
    return [guard_tool(t, **kwargs) for t in tools]
