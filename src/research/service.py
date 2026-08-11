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
    model = ChatOpenAI(
        model=resolved.model,
        base_url=resolved.base_url,
        api_key=resolved.api_key,
        temperature=0,
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
        chapter_max_claims=resolved.chapter_max_claims,
        chapter_max_decisions=resolved.chapter_max_decisions,
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
