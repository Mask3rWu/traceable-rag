from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.config import ResearchModelConfig
from src.research.client import OpenAIResearchModel, _parse_json_object
from src.research.models import Evidence


class _FakeCompletions:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(payload, ensure_ascii=False)
                    )
                )
            ]
        )


class _FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class ResearchClientTest(unittest.TestCase):
    def _model(self, responses: list[dict]):
        client = _FakeClient(responses)
        config = ResearchModelConfig(
            model="deepseek-test",
            base_url="https://example.test",
            api_key="secret",
        )
        return OpenAIResearchModel(config, client=client), client

    def test_plans_unique_trimmed_queries_with_json_mode(self):
        model, client = self._model(
            [{"queries": ["  第一方面  ", "第一方面", "第二方面"]}]
        )

        queries = model.plan_queries("问题", max_queries=2)

        self.assertEqual(queries, ["第一方面", "第二方面"])
        call = client.chat.completions.calls[0]
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertEqual(call["temperature"], 0)

    def test_synthesizes_structured_claim_with_exact_citation(self):
        model, _ = self._model(
            [
                {
                    "summary": "结论摘要",
                    "claims": [
                        {
                            "claim_id": "cl-1",
                            "text": "装甲破裂降低防护能力。",
                            "conclusion_type": "direct",
                            "citations": [
                                {"evidence_id": "ev-1", "quote": "降低防护能力"}
                            ],
                        }
                    ],
                    "conflicts": [],
                }
            ]
        )
        evidence = Evidence(
            evidence_id="ev-1",
            chunk_id="chunk-1",
            content_hash="a" * 64,
            document_id="doc",
            source_file="doc.pdf",
            page_start=1,
            page_end=1,
            block_ids=["block-1"],
            quote="装甲破裂会降低防护能力。",
        )

        draft = model.synthesize("问题", [evidence])

        self.assertEqual(draft.claims[0].citations[0].evidence_id, "ev-1")

    def test_rejects_empty_draft(self):
        model, _ = self._model([{"summary": "", "claims": [], "conflicts": []}])
        with self.assertRaises(ValueError):
            model.synthesize("问题", [])

    def test_parses_fenced_json_and_rejects_arrays(self):
        self.assertEqual(_parse_json_object('```json\n{"ok": true}\n```'), {"ok": True})
        with self.assertRaisesRegex(ValueError, "JSON object"):
            _parse_json_object("[]")
