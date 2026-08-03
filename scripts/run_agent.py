"""Run the unified routed LangGraph research agent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.agent_store import AgentRunStore  # noqa: E402
from src.research.service import build_research_agent  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route a request to the fast or supervisor research agent"
    )
    parser.add_argument("request", help="Question or research deliverable request")
    parser.add_argument(
        "--runs-dir", type=Path, help="Override the agent run output directory"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = AgentRunStore(args.runs_dir)
    agent = build_research_agent(store=store)
    run, path = agent.run(args.request)
    print(run.answer.content)
    if run.answer.limitations:
        print("\nLimitations:")
        for limitation in run.answer.limitations:
            print(f"- {limitation}")
    if run.answer.evidence_ids:
        evidence_by_id = {item.evidence_id: item for item in run.evidence}
        print("\nSources:")
        for evidence_id in run.answer.evidence_ids:
            evidence = evidence_by_id[evidence_id]
            pages = (
                str(evidence.page_start)
                if evidence.page_start == evidence.page_end
                else f"{evidence.page_start}-{evidence.page_end}"
            )
            print(f"- [{evidence_id}] {evidence.source_file}, pages {pages}")
    print(f"\nroute: {run.route.mode} ({run.route.reason})")
    print(f"evidence: {len(run.evidence)}, worker_packets: {len(run.worker_packets)}")
    print(f"run_json: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
