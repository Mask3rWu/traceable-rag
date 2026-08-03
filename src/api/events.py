"""Compact LangChain callback events for the web client."""
from __future__ import annotations

import threading
from typing import Any, Callable
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


def _name(serialized: dict[str, Any] | None, fallback: str) -> str:
    if serialized:
        return str(serialized.get("name") or serialized.get("id", [fallback])[-1])
    return fallback


def _preview(value: Any, *, limit: int = 240) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


class AgentEventCallback(BaseCallbackHandler):
    def __init__(self, emit: Callable[[str, dict[str, Any]], None]) -> None:
        self.emit = emit
        self._chains: dict[UUID, str] = {}
        self._lock = threading.Lock()

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        name = str(kwargs.get("name") or _name(serialized, "chain"))
        run_id = kwargs.get("run_id")
        if isinstance(run_id, UUID):
            with self._lock:
                self._chains[run_id] = name
        visible = {
            "research-router",
            "router",
            "fast-react-agent",
            "research-supervisor",
            "research-worker",
        }
        if name in visible:
            self.emit("stage_started", {"stage": name})

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        with self._lock:
            name = self._chains.pop(run_id, "chain")
        if name == "router" and isinstance(outputs, dict):
            route = outputs.get("route")
            if isinstance(route, dict):
                self.emit("route_selected", route)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        with self._lock:
            self._chains.pop(run_id, None)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        name = _name(serialized, "tool")
        inputs = kwargs.get("inputs")
        data: dict[str, Any] = {"tool": name}
        if isinstance(inputs, dict):
            for key in ("query", "task"):
                if key in inputs:
                    data[key] = _preview(inputs[key])
        elif input_str:
            data["input"] = _preview(input_str)
        self.emit("tool_started", data)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self.emit("tool_completed", {"output": _preview(output, limit=120)})

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self.emit("tool_failed", {"error": _preview(error)})
