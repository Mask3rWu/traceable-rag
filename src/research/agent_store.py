"""Atomic persistence for routed agent results."""
from __future__ import annotations

import os
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
