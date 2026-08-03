"""Durable JSON persistence for inspectable research runs."""
from __future__ import annotations

import os
from pathlib import Path

from src.paths import PROCESSED_ROOT
from src.research.models import ResearchRun


class ResearchRunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROCESSED_ROOT / "research" / "runs"

    def path_for(self, run_id: str) -> Path:
        return self.root / run_id / "run.json"

    def save(self, run: ResearchRun) -> Path:
        run.touch()
        path = self.path_for(run.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            run.model_dump_json(indent=2), encoding="utf-8", newline="\n"
        )
        os.replace(temporary, path)
        return path

    def load(self, run_id: str) -> ResearchRun:
        path = self.path_for(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Research run not found: {run_id}")
        return ResearchRun.model_validate_json(path.read_text(encoding="utf-8"))
