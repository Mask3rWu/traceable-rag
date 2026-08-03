"""Atomic persistence for routed agent results."""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from src.paths import PROCESSED_ROOT
from src.research.agent_models import AgentRun


class AgentRunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROCESSED_ROOT / "research" / "agent-runs"

    def path_for(self, run_id: str) -> Path:
        return self.root / run_id / "run.json"

    def save(self, run: AgentRun) -> Path:
        path = self.path_for(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            run.model_dump_json(indent=2), encoding="utf-8", newline="\n"
        )
        os.replace(temporary, path)
        return path

    def load(self, run_id: str) -> AgentRun:
        path = self.path_for(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Agent run not found: {run_id}")
        return AgentRun.model_validate_json(path.read_text(encoding="utf-8"))

    def iter_runs(self) -> Iterator[AgentRun]:
        if not self.root.is_dir():
            return
        paths = sorted(
            self.root.glob("*/run.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                yield AgentRun.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue

    def list(self, *, limit: int = 50) -> list[AgentRun]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        runs: list[AgentRun] = []
        for run in self.iter_runs():
            runs.append(run)
            if len(runs) >= limit:
                break
        return runs
