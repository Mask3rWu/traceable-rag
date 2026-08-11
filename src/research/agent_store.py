"""Atomic persistence for routed agent results."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from src.paths import PROCESSED_ROOT
from src.research.agent_models import AgentRun, RunCheckpoint


class AgentRunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROCESSED_ROOT / "research" / "agent-runs"

    def path_for(self, run_id: str) -> Path:
        return self.root / run_id / "run.json"

    def checkpoint_path_for(self, run_id: str) -> Path:
        return self.root / run_id / "checkpoint.json"

    def metrics_path_for(self, run_id: str) -> Path:
        return self.root / run_id / "metrics.json"

    def save_metrics(self, run_id: str, metrics: dict) -> Path:
        return self._atomic_write(
            self.metrics_path_for(run_id),
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        )

    def load_metrics(self, run_id: str) -> dict | None:
        path = self.metrics_path_for(run_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
        return path

    def save(self, run: AgentRun) -> Path:
        path = self.path_for(run.run_id)
        return self._atomic_write(path, run.model_dump_json(indent=2))

    def save_checkpoint(self, checkpoint: RunCheckpoint) -> Path:
        return self._atomic_write(
            self.checkpoint_path_for(checkpoint.run_id),
            checkpoint.model_dump_json(indent=2),
        )

    def load_checkpoint(self, run_id: str) -> RunCheckpoint:
        path = self.checkpoint_path_for(run_id)
        if not path.is_file():
            run = self.load(run_id)
            if run.document_plan is None:
                raise FileNotFoundError(f"Run has no resumable document plan: {run_id}")
            return RunCheckpoint(
                run_id=run.run_id,
                request=run.request,
                route=run.route,
                document_plan=run.document_plan,
                evidence=run.evidence,
                worker_packets=run.worker_packets,
                evidence_aliases=run.evidence_aliases,
                parent_run_id=run.parent_run_id,
                attempt=run.attempt,
                created_at=run.created_at,
            )
        return RunCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))

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
