"""Run one traceable research question and persist its complete state."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ResearchModelConfig  # noqa: E402
from src.research.client import OpenAIResearchModel  # noqa: E402
from src.research.evidence import CitationVerifier, EvidenceResolver  # noqa: E402
from src.research.store import ResearchRunStore  # noqa: E402
from src.research.workflow import ResearchWorkflow  # noqa: E402
from src.retrieval.catalog import ChunkCatalog  # noqa: E402
from src.retrieval.service import RetrievalService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed research workflow over the local knowledge base"
    )
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Override processed/research/runs output directory",
    )
    parser.add_argument("--max-queries", type=int, help="Override RESEARCH_MAX_QUERIES")
    parser.add_argument(
        "--evidence-limit",
        type=int,
        help="Override RESEARCH_EVIDENCE_LIMIT for each query",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ResearchModelConfig.from_env()
    max_queries = args.max_queries or config.max_queries
    evidence_limit = args.evidence_limit or config.evidence_limit
    if max_queries <= 0 or evidence_limit <= 0:
        raise SystemExit("--max-queries and --evidence-limit must be greater than zero")

    catalog = ChunkCatalog.load()
    store = ResearchRunStore(args.runs_dir)
    workflow = ResearchWorkflow(
        model=OpenAIResearchModel(config),
        retrieval=RetrievalService(),
        resolver=EvidenceResolver(catalog),
        verifier=CitationVerifier(catalog),
        store=store,
        max_queries=max_queries,
        evidence_limit=evidence_limit,
    )
    run = workflow.run(args.question)

    print(run.summary)
    print(f"\nrun_id: {run.run_id}")
    print(f"run_json: {store.path_for(run.run_id).resolve()}")
    print(f"claims: {len(run.claims)}, evidence: {len(run.evidence)}, conflicts: {len(run.conflicts)}")
    print("\nVerified claims:")
    for claim in run.claims:
        citations = ", ".join(
            sorted({citation.evidence_id for citation in claim.citations})
        )
        print(
            f"- [{claim.conclusion_type}] {claim.text} "
            f"(sources: {citations})"
        )
    print("\nSources:")
    for evidence in run.evidence:
        pages = (
            str(evidence.page_start)
            if evidence.page_start == evidence.page_end
            else f"{evidence.page_start}-{evidence.page_end}"
        )
        section = " > ".join(evidence.section_path) or "(no section)"
        print(
            f"[{evidence.evidence_id}] {evidence.source_file}, "
            f"pages {pages}, {section}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
