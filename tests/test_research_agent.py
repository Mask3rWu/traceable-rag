from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import ResearchModelConfig
from src.research.service import build_research_agent


class BuildResearchAgentTest(unittest.TestCase):
    def _config(self, **overrides) -> ResearchModelConfig:
        base = {
            "model": "test-model",
            "base_url": "https://research.example/v1",
            "api_key": "test-secret",
        }
        base.update(overrides)
        return ResearchModelConfig(**base)

    def test_wires_per_role_token_budgets_and_thinking_via_extra_body(self):
        """三个 supervisor 侧模型各自带独立的 token_budget/thinking，均走
        extra_body.max_tokens / extra_body.thinking（deepseek 只认 max_tokens）。
        thinking=false=关（发送 disabled）；thinking=true=开（不发送字段）。"""
        cfg = self._config(
            fast_token_budget=900,
            planner_token_budget=300,
            worker_token_budget=500,
            reviewer_token_budget=700,
            thinking_fast=False,
            thinking_planner=False,
            thinking_worker=True,
            thinking_reviewer=False,
        )
        with (
            patch("src.research.service.ChunkCatalog.load", return_value=object()),
            patch("src.research.service.RetrievalService"),
            patch("src.research.service.AgentRuntime") as runtime_cls,
        ):
            build_research_agent(cfg)

        kwargs = runtime_cls.call_args.kwargs
        fast = kwargs["model"]
        planner = kwargs["planner_model"]
        worker = kwargs["worker_model"]
        reviewer = kwargs["reviewer_model"]

        self.assertEqual(fast.extra_body["max_tokens"], 900)
        self.assertEqual(planner.extra_body["max_tokens"], 300)
        self.assertEqual(worker.extra_body["max_tokens"], 500)
        self.assertEqual(reviewer.extra_body["max_tokens"], 700)

        self.assertEqual(fast.extra_body["thinking"], {"type": "disabled"})
        self.assertEqual(planner.extra_body["thinking"], {"type": "disabled"})
        self.assertNotIn("thinking", worker.extra_body)
        self.assertEqual(reviewer.extra_body["thinking"], {"type": "disabled"})

    def test_wired_models_never_set_field_level_max_tokens(self):
        """预算必须只经 extra_body.max_tokens 生效；字段级 max_tokens 必须保持
        None，否则 langchain 会改写成 max_completion_tokens 而被 deepseek 忽略。"""
        cfg = self._config(
            fast_token_budget=900,
            planner_token_budget=300,
            worker_token_budget=500,
            reviewer_token_budget=700,
        )
        with (
            patch("src.research.service.ChunkCatalog.load", return_value=object()),
            patch("src.research.service.RetrievalService"),
            patch("src.research.service.AgentRuntime") as runtime_cls,
        ):
            build_research_agent(cfg)

        kwargs = runtime_cls.call_args.kwargs
        for role in ("model", "planner_model", "worker_model", "reviewer_model"):
            model = kwargs[role]
            self.assertIsNone(model.max_tokens, f"{role} must not set field max_tokens")

    def test_unset_budgets_leave_extra_body_without_max_tokens(self):
        """预算留空（None）时 extra_body 不含 max_tokens，维持不限的现状。"""
        cfg = self._config(
            fast_token_budget=None,
            planner_token_budget=None,
            worker_token_budget=None,
            reviewer_token_budget=None,
        )
        with (
            patch("src.research.service.ChunkCatalog.load", return_value=object()),
            patch("src.research.service.RetrievalService"),
            patch("src.research.service.AgentRuntime") as runtime_cls,
        ):
            build_research_agent(cfg)

        kwargs = runtime_cls.call_args.kwargs
        for role in ("model", "planner_model", "worker_model", "reviewer_model"):
            model = kwargs[role]
            # 预算全 None => build_chat_model 不设 extra_body（为 None），自然无上限。
            self.assertIsNone(
                (model.extra_body or {}).get("max_tokens"),
                f"{role} budget should be unset",
            )
            self.assertIsNone(model.max_tokens)


if __name__ == "__main__":
    unittest.main()