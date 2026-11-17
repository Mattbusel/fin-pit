"""Guard LangChain and LangGraph tools: ``guard_tool(tool)``.

``guard_tool`` returns a new tool with the same name, description, argument schema and
response format whose calls go through ``fin_pit.guard``: date arguments later than the
active ``asof`` raise ``LookaheadError`` before the vendor is called, and the result (and a
``ToolMessage`` artifact) is filtered or checked on the way out. LangGraph's ``ToolNode``
takes LangChain tools, so the guarded tool drops into a graph unchanged.

langchain-core is only imported when a LangChain tool is passed; a plain callable is wrapped
with ``fin_pit.guard`` directly.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple

from .guard import DEFAULT_DATE_ARGS, LookaheadError, _Guard, _GuardOptions, guard

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


def _tool_call_id_fields(tool: Any) -> Tuple[str, ...]:
    """Argument names annotated ``InjectedToolCallId`` in ``tool.args_schema``."""
    schema = getattr(tool, "args_schema", None)
    if schema is None or isinstance(schema, dict):
        return ()
    try:
        from langchain_core.tools import InjectedToolCallId
        from langchain_core.tools.base import _is_injected_arg_type, get_all_basemodel_annotations

        return tuple(
            k
            for k, v in get_all_basemodel_annotations(schema).items()
            if _is_injected_arg_type(v, injected_type=InjectedToolCallId)
        )
    except Exception:  # pragma: no cover - older or newer langchain-core layouts
        return ()


def guard_tool(
    tool: Any,
    *,
    mode: Optional[str] = None,
    unstamped: Optional[str] = None,
    column: Optional[str] = None,
    date_args: Optional[Iterable[str]] = DEFAULT_DATE_ARGS,
    date_tz: Any = None,
    scan_text: bool = False,
    require_asof: Optional[bool] = None,
    unparsed: str = "warn",
) -> Any:
    """Wrap a LangChain ``BaseTool`` (or any callable) in the point-in-time guard.

    Keyword arguments are those of ``fin_pit.guard``. For a ``BaseTool`` the wrapper is a
    ``StructuredTool`` that keeps ``name``, ``description``, ``args_schema``,
    ``response_format``, ``return_direct``, ``handle_tool_error`` and ``tags``. Its sync and
    async paths call the original tool's ``invoke``/``ainvoke`` with the run's config (so
    ``RunnableConfig`` arguments and callbacks still work), and with a full ``ToolCall``
    when the original needs one (an ``InjectedToolCallId`` argument, or
    ``response_format="content_and_artifact"``). The artifact of a ``ToolMessage`` is
    checked or filtered like any other output; a LangGraph ``Command`` is passed through.
    """
    opts = _GuardOptions(mode, unstamped, column, date_args, date_tz, scan_text, require_asof, unparsed)
    try:
        from langchain_core.tools import BaseTool, StructuredTool
    except ImportError:
        BaseTool = None  # type: ignore[assignment,misc]

    if BaseTool is None or not isinstance(tool, BaseTool):
        if not callable(tool):
            raise TypeError(f"guard_tool needs a LangChain tool or a callable, got {type(tool).__name__}")
        return guard(
            tool,
            mode=mode,
            unstamped=unstamped,
            column=column,
            date_args=date_args,
            date_tz=date_tz,
            scan_text=scan_text,
            require_asof=require_asof,
            unparsed=unparsed,
        )

    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.runnables.config import patch_config
    from langchain_core.tools import ToolException

    original = tool
    error_cls = _tool_error_class(ToolException)
    id_fields = _tool_call_id_fields(original)
    with_artifact = getattr(original, "response_format", "content") == "content_and_artifact"
    checker = _Guard(original, opts, name=str(original.name))

    def _prepare(kwargs: dict, config: Any, callbacks: Any) -> Tuple[Any, Any, bool]:
        """(input for the original, config, unwrap?), with the ToolCall rebuilt if needed."""
        cfg = patch_config(config, callbacks=callbacks) if config is not None else None
        args = dict(kwargs)
        call_id = None
        for k in id_fields:
            if k in args:
                call_id = args.pop(k)
        if call_id is None and not with_artifact:
            return args, cfg, False
        tool_call = {
            "type": "tool_call",
            "name": original.name,
            "args": args,
            "id": call_id if call_id is not None else f"fin-pit-{uuid.uuid4().hex}",
        }
        return tool_call, cfg, call_id is None

    def _check_message(msg: Any, state: Any) -> Any:
        update = {}
        if msg.artifact is not None:
            art = checker.after(msg.artifact, state)
            if art is not msg.artifact:
                update["artifact"] = art
        if isinstance(msg.content, list):
            content = checker.after(msg.content, state)
            if content is not msg.content:
                update["content"] = content
        elif isinstance(msg.content, str) and opts.scan_text and state is not None:
            checker.scan(msg.content, state)
        return msg.model_copy(update=update) if update else msg

    def _finish(out: Any, state: Any, unwrap: bool) -> Any:
        if isinstance(out, ToolMessage):
            out = _check_message(out, state)
            if unwrap:
                if out.status == "error":
                    raise ToolException(out.content)
                return out.content, out.artifact
            return (out, None) if with_artifact else out
        if type(out).__name__ == "Command":
            # LangGraph Command (state updates, handoffs): nothing to filter.
            return (out, None) if with_artifact else out
        return checker.after(out, state)

    def sync_fn(config: Any = None, callbacks: Any = None, **kwargs: Any) -> Any:
        try:
            state = checker.before((), kwargs)
            payload, cfg, unwrap = _prepare(kwargs, config, callbacks)
            return _finish(original.invoke(payload, cfg), state, unwrap)
        except LookaheadError as exc:
            raise _as_tool_error(exc, error_cls) from exc

    async def async_fn(config: Any = None, callbacks: Any = None, **kwargs: Any) -> Any:
        try:
            state = checker.before((), kwargs)
            payload, cfg, unwrap = _prepare(kwargs, config, callbacks)
            return _finish(await original.ainvoke(payload, cfg), state, unwrap)
        except LookaheadError as exc:
            raise _as_tool_error(exc, error_cls) from exc

    # Real annotations (not strings), so StructuredTool hands over the run's config.
    sync_fn.__annotations__ = {"config": RunnableConfig, "callbacks": Any, "kwargs": Any, "return": Any}
    async_fn.__annotations__ = dict(sync_fn.__annotations__)
    sync_fn.__name__ = async_fn.__name__ = str(original.name)

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
        response_format="content_and_artifact" if with_artifact else "content",
        **extra,
    )
    return wrapped


def guard_tools(tools: Iterable[Any], **kwargs: Any) -> List[Any]:
    """``guard_tool`` over a list (for ``create_react_agent(model, guard_tools(tools))``)."""
    return [guard_tool(t, **kwargs) for t in tools]
