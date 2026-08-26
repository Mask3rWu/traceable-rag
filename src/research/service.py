"""Composition root for the routed research agent."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langchain_core.callbacks import BaseCallbackHandler
from langfuse.langchain import CallbackHandler

from src.config import ResearchModelConfig
from src.research.agent_models import AgentRun
from src.research.agent_store import AgentRunStore
from src.research.evidence import EvidenceResolver
from src.research.eval_metrics import RuntimeMetrics
from src.research.graph import AgentRuntime
from src.research.tools import EvidenceWorkspace
from src.retrieval.catalog import ChunkCatalog
from src.retrieval.service import RetrievalService


@dataclass
class RoutedResearchAgent:
    runtime: AgentRuntime
    config: ResearchModelConfig
    langfuse: Langfuse | None = None

    def attach_metrics(self, metrics: RuntimeMetrics | None) -> None:
        """Wire a per-run :class:`RuntimeMetrics` into the runtime's schema hooks."""
        self.runtime.attach_metrics(metrics)

    def run(
        self,
        request: str,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        route_guard: Callable[[str], bool] | None = None,
    ) -> tuple[AgentRun, Any]:
        resolved_callbacks = list(callbacks or [])
        if self.langfuse is not None:
            resolved_trace_id = trace_id or self.langfuse.create_trace_id(seed=run_id)
            resolved_callbacks.append(
                CallbackHandler(
                    public_key=self.config.langfuse_public_key,
                    update_trace=True,
                    trace_context={"trace_id": resolved_trace_id},
                )
            )
            trace_id = resolved_trace_id
        runnable_config = {
            "callbacks": resolved_callbacks,
            "run_name": "research-router",
            "tags": ["research-agent"],
            "metadata": {"entrypoint": "routed-research-agent"},
            "recursion_limit": self.config.max_steps * 4 + 8,
        }
        try:
            return self.runtime.run(
                request,
                config=runnable_config,
                run_id=run_id,
                trace_id=trace_id,
                cancel_check=cancel_check,
                route_guard=route_guard,
            )
        finally:
            if self.langfuse is not None:
                self.langfuse.flush()

    def resume(
        self,
        checkpoint,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        start_chapter: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[AgentRun, Any]:
        resolved_callbacks = list(callbacks or [])
        runnable_config = {
            "callbacks": resolved_callbacks,
            "run_name": "research-router-resume",
            "tags": ["research-agent", "resume"],
            "metadata": {"entrypoint": "routed-research-agent"},
            "recursion_limit": self.config.max_steps * 4 + 8,
        }
        return self.runtime.resume(
            checkpoint,
            config=runnable_config,
            run_id=run_id,
            trace_id=trace_id,
            start_chapter=start_chapter,
            cancel_check=cancel_check,
        )


def build_research_agent(
    config: ResearchModelConfig | None = None,
    *,
    store: AgentRunStore | None = None,
    metrics: RuntimeMetrics | None = None,
) -> RoutedResearchAgent:
    resolved = config or ResearchModelConfig.from_env()

    def _chat_model(*, disable_thinking: bool) -> ChatOpenAI:
        kwargs = {
            "model": resolved.model,
            "base_url": resolved.base_url,
            "api_key": resolved.api_key,
            "temperature": 0,
        }
        if disable_thinking:
            # OpenAI 兼容的顶层字段。必须走 extra_body：model_kwargs 会被当作
            # create() 具名参数而触发 SDK 校验错误；extra_body 由 SDK 合并进请求
            # 体，经探测确认 deepseek 认它并真正关掉思考。
            # （注意：langchain 会把 max_tokens 自发转成 max_completion_tokens，
            # deepseek 不认后者，故限 token 不能走 ChatOpenAI.max_tokens。）
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return ChatOpenAI(**kwargs)

    model = _chat_model(disable_thinking=resolved.disable_thinking_fast)
    worker_model = _chat_model(
        disable_thinking=resolved.disable_thinking_supervisor
    )
    catalog = ChunkCatalog.load()
    workspace = EvidenceWorkspace(
        retrieval=RetrievalService(),
        resolver=EvidenceResolver(catalog),
        default_top_k=resolved.retrieval_top_k,
        max_evidence_reads=resolved.max_evidence_reads,
    )
    runtime = AgentRuntime(
        model=model,
        worker_model=worker_model,
        workspace=workspace,
        store=store,
        max_steps=resolved.max_steps,
        fast_max_steps=resolved.fast_max_steps,
        worker_max_steps=resolved.worker_max_steps,
        supervisor_max_steps=resolved.supervisor_max_steps,
        max_workers=resolved.max_workers,
        max_subtasks=resolved.max_subtasks,
        document_max_chars=resolved.document_max_chars,
        chapter_max_chars=resolved.chapter_max_chars,
        chapter_max_rules=resolved.chapter_max_rules,
        max_search_per_worker=resolved.max_search_per_worker,
        metrics=metrics,
    )
    langfuse = None
    if resolved.langfuse_enabled:
        langfuse = Langfuse(
            public_key=resolved.langfuse_public_key,
            secret_key=resolved.langfuse_secret_key,
            base_url=resolved.langfuse_base_url,
        )
    return RoutedResearchAgent(runtime=runtime, config=resolved, langfuse=langfuse)
