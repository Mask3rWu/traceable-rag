"""Composition root for the routed research agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config import ResearchModelConfig
from src.research.agent_models import AgentRun
from src.research.agent_store import AgentRunStore
from src.research.evidence import CitationVerifier, EvidenceResolver
from src.research.graph import AgentRuntime
from src.research.tools import EvidenceWorkspace
from src.retrieval.catalog import ChunkCatalog
from src.retrieval.service import RetrievalService


@dataclass
class RoutedResearchAgent:
    runtime: AgentRuntime
    config: ResearchModelConfig
    langfuse: Langfuse | None = None

    def run(self, request: str) -> tuple[AgentRun, Any]:
        callbacks = []
        if self.langfuse is not None:
            callbacks.append(
                CallbackHandler(
                    public_key=self.config.langfuse_public_key, update_trace=True
                )
            )
        runnable_config = {
            "callbacks": callbacks,
            "run_name": "research-router",
            "tags": ["research-agent"],
            "metadata": {"entrypoint": "routed-research-agent"},
            "recursion_limit": self.config.max_steps * 4 + 8,
        }
        try:
            return self.runtime.run(request, config=runnable_config)
        finally:
            if self.langfuse is not None:
                self.langfuse.flush()


def build_research_agent(
    config: ResearchModelConfig | None = None,
    *,
    store: AgentRunStore | None = None,
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
        verifier=CitationVerifier(catalog),
        default_top_k=resolved.retrieval_top_k,
        max_evidence_reads=resolved.max_evidence_reads,
    )
    runtime = AgentRuntime(
        model=model,
        workspace=workspace,
        store=store,
        max_steps=resolved.max_steps,
        max_workers=resolved.max_workers,
        max_subtasks=resolved.max_subtasks,
    )
    langfuse = None
    if resolved.langfuse_enabled:
        langfuse = Langfuse(
            public_key=resolved.langfuse_public_key,
            secret_key=resolved.langfuse_secret_key,
            base_url=resolved.langfuse_base_url,
        )
    return RoutedResearchAgent(runtime=runtime, config=resolved, langfuse=langfuse)
