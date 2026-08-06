from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.research.client import ResearchDraft
from src.research.evidence import EvidenceResolver
from src.research.models import Citation, Claim, Conflict
from src.research.store import ResearchRunStore
from src.research.workflow import ResearchWorkflow
from src.retrieval.catalog import ChunkCatalog
from src.retrieval.contracts import SearchResult
from src.retrieval.indexing import SourceChunk
from src.schema import Chunk


class _FakeRetrieval:
    def __init__(self, result: SearchResult) -> None:
        self.result = result
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int):
        self.queries.append(query)
        return [self.result]


class _FakeModel:
    def __init__(self, evidence_id: str) -> None:
        self.evidence_id = evidence_id

    def plan_queries(self, question: str, *, max_queries: int) -> list[str]:
        return [question, "结构防护 装甲破裂", "毁伤等级"][:max_queries]

    def synthesize(self, question, evidence):
        del evidence
        return ResearchDraft(
            summary="现有证据表明装甲破裂会降低结构防护能力。",
            claims=[
                Claim(
                    claim_id="cl-1",
                    text="装甲破裂会降低结构防护能力。",
                    conclusion_type="direct",
                    citations=[Citation(evidence_id=self.evidence_id)],
                )
            ],
            conflicts=[
                Conflict(
                    conflict_id="co-1",
                    description="不同来源可能采用不同毁伤等级。",
                    claim_ids=["cl-1"],
                    evidence_ids=[self.evidence_id],
                )
            ],
        )


class ResearchWorkflowTest(unittest.TestCase):
    def _build(self, root: Path):
        source = SourceChunk(
            chunk=Chunk(
                chunk_id="doc_C0001",
                document_id="doc",
                text="装甲破裂会降低结构防护能力。",
                embedding_text="装甲破裂会降低结构防护能力。",
                block_ids=["doc_P003_B01"],
                page_start=3,
                page_end=3,
                section_path=["3.1"],
                source_file="标准.pdf",
            ),
            content_hash="a" * 64,
        )
        catalog = ChunkCatalog([source])
        result = SearchResult(
            chunk_id=source.chunk.chunk_id,
            content_hash=source.content_hash,
            bm25_rank=1,
            bm25_score=9.0,
            final_rank=1,
        )
        evidence_id = EvidenceResolver(catalog).resolve("x", result).evidence_id
        retrieval = _FakeRetrieval(result)
        store = ResearchRunStore(root)
        workflow = ResearchWorkflow(
            model=_FakeModel(evidence_id),
            retrieval=retrieval,
            resolver=EvidenceResolver(catalog),
            store=store,
            max_queries=3,
            evidence_limit=5,
        )
        return workflow, retrieval, store

    def test_completes_traceable_multi_query_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow, retrieval, store = self._build(Path(tmp))

            run = workflow.run("装甲破裂有什么影响？")
            persisted = store.load(run.run_id)

        self.assertEqual(run.status, "completed")
        self.assertEqual(retrieval.queries, ["装甲破裂有什么影响？", "结构防护 装甲破裂", "毁伤等级"])
        self.assertEqual(len(run.evidence), 1)
        self.assertEqual(len(run.evidence[0].retrieval), 3)
        self.assertEqual([call.tool for call in run.tool_calls], ["plan_queries", "search", "search", "search", "synthesize"])
        self.assertEqual(persisted.status, "completed")

    def test_rejects_duplicate_claim_ids(self):
        claims = [
            Claim(
                claim_id="same",
                text="第一条",
                conclusion_type="direct",
                citations=[Citation(evidence_id="ev-1")],
            ),
            Claim(
                claim_id="same",
                text="第二条",
                conclusion_type="direct",
                citations=[Citation(evidence_id="ev-1")],
            ),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate claim IDs"):
            ResearchWorkflow._validate_draft(claims, [])
