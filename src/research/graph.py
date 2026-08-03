"""Routed LangGraph ReAct runtime with controlled research delegation."""
from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.research.agent_models import AgentAnswer, AgentRun, ResearchPacket, RouteDecision
from src.research.agent_store import AgentRunStore
from src.research.tools import EvidenceWorkspace


FAST_PROMPT = """You are a traceable knowledge-base question-answering agent.
Rewrite the user's need into focused searches. Use search_knowledge, inspect previews, and
read only the evidence needed. You may search again when coverage is weak. Answer only from
retrieved evidence. Cite evidence IDs inline and finish by calling submit_answer."""

WORKER_PROMPT = """You are a research worker handling one bounded task. Use iterative
search_knowledge and read_evidence calls until the acceptance criteria are met or a concrete
evidence gap is established. Never invent facts. Exact citation quotes must be substrings of
the evidence. Finish by calling submit_research with a compact structured packet."""

SUPERVISOR_PROMPT = """You are a research supervisor. Decompose complex requests into
independent, bounded research tasks and call delegate_research for them. Workers return
compact research packets with evidence IDs, exact quotes, conflicts, and gaps. Review coverage
and delegate follow-up tasks when necessary. Synthesize the final deliverable without inventing
evidence, cite evidence IDs inline, state limitations, and finish with submit_answer.
You may submit a substantive deliverable only from sufficient worker packets with verified evidence.
Only you may delegate; workers cannot create more workers."""

ROUTER_PROMPT = """Classify the execution mode for a knowledge-base request. Choose fast
for a focused question answerable with a few searches. Choose supervisor for requests requiring
multi-part research, cross-source comparison, an assessment standard, report, taxonomy, rules,
or another substantial structured deliverable. Return a JSON object matching this schema:
{"mode": "fast|supervisor", "reason": "short operational reason"}."""


class ReactState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    result: dict[str, Any]


class RootState(TypedDict, total=False):
    request: str
    route: dict[str, Any]
    answer: dict[str, Any]


class DelegateInput(BaseModel):
    task: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)


class AgentRuntime:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        workspace: EvidenceWorkspace,
        store: AgentRunStore | None = None,
        max_steps: int = 12,
        max_workers: int = 4,
        max_subtasks: int = 8,
    ) -> None:
        if min(max_steps, max_workers, max_subtasks) <= 0:
            raise ValueError("Agent budgets must be greater than zero")
        self.model = model
        self.workspace = workspace
        self.store = store or AgentRunStore()
        self.max_steps = max_steps
        self.max_workers = max_workers
        self.max_subtasks = max_subtasks
        self._packets: list[ResearchPacket] = []
        self._subtask_count = 0
        self._lock = threading.RLock()
        self._run_lock = threading.Lock()
        self._fast_graph = self._build_fast_graph()
        self._supervisor_graph = self._build_supervisor_graph()
        self.graph = self._build_root_graph()

    @property
    def packets(self) -> list[ResearchPacket]:
        with self._lock:
            return list(self._packets)

    @staticmethod
    def _submit_answer_tool(
        validator: Callable[[AgentAnswer], None] | None = None,
    ) -> BaseTool:
        @tool(args_schema=AgentAnswer)
        def submit_answer(
            content: str, evidence_ids: list[str], limitations: list[str]
        ) -> str:
            """Submit the final answer and stop this agent."""

            answer = AgentAnswer(
                content=content, evidence_ids=evidence_ids, limitations=limitations
            )
            if validator is not None:
                validator(answer)
            return "submitted"

        return submit_answer

    @staticmethod
    def _submit_research_tool() -> BaseTool:
        @tool(args_schema=ResearchPacket)
        def submit_research(
            task: str,
            status: Literal["sufficient", "insufficient"],
            summary: str,
            claims: list,
            conflicts: list,
            gaps: list[str],
            evidence_ids: list[str],
        ) -> str:
            """Submit a structured research packet and stop this worker."""

            return "submitted"

        return submit_research

    def _build_react_graph(
        self,
        *,
        prompt: str,
        tools: Sequence[BaseTool],
        submit_name: str,
        result_model: type[BaseModel],
        exhausted_result: Callable[[], BaseModel],
        can_submit: Callable[[dict[str, Any]], bool] | None = None,
        graph_name: str,
    ):
        bound_model = self.model.bind_tools(list(tools))
        tool_node = ToolNode(tools, handle_tool_errors=True)

        def call_model(state: ReactState, config: RunnableConfig) -> dict:
            response = bound_model.invoke(
                [SystemMessage(content=prompt), *state["messages"]], config=config
            )
            return {"messages": [response], "steps": state.get("steps", 0) + 1}

        def next_step(state: ReactState) -> str:
            message = state["messages"][-1]
            calls = message.tool_calls if isinstance(message, AIMessage) else []
            submission = calls[0] if len(calls) == 1 else None
            submission_allowed = (
                submission is not None
                and submission["name"] == submit_name
                and (can_submit is None or can_submit(submission["args"]))
            )
            if (
                submission is not None
                and submission["name"] == submit_name
                and submission_allowed
            ):
                return "submit"
            if state.get("steps", 0) >= self.max_steps:
                return "exhausted"
            return "tools" if calls else "exhausted"

        def submit(state: ReactState) -> dict:
            message = state["messages"][-1]
            call = next(
                item for item in message.tool_calls if item["name"] == submit_name
            )
            result = result_model.model_validate(call["args"])
            return {"result": result.model_dump(mode="json")}

        def exhausted(_: ReactState) -> dict:
            result = exhausted_result()
            return {"result": result.model_dump(mode="json")}

        builder = StateGraph(ReactState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", tool_node)
        builder.add_node("submit", submit)
        builder.add_node("exhausted", exhausted)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            next_step,
            {"tools": "tools", "submit": "submit", "exhausted": "exhausted"},
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("submit", END)
        builder.add_edge("exhausted", END)
        return builder.compile(name=graph_name)

    def _build_fast_graph(self):
        tools = self.workspace.make_retrieval_tools()
        tools.append(self._submit_answer_tool(self._validate_fast_answer))
        return self._build_react_graph(
            prompt=FAST_PROMPT,
            tools=tools,
            submit_name="submit_answer",
            result_model=AgentAnswer,
            exhausted_result=lambda: AgentAnswer(
                content="未能在执行预算内形成可靠答案。",
                limitations=["Agent step budget exhausted"],
            ),
            can_submit=self._can_submit_fast_answer,
            graph_name="fast-react-agent",
        )

    def _build_worker_graph(self):
        tools = [*self.workspace.make_retrieval_tools(), self._submit_research_tool()]
        return self._build_react_graph(
            prompt=WORKER_PROMPT,
            tools=tools,
            submit_name="submit_research",
            result_model=ResearchPacket,
            exhausted_result=lambda: ResearchPacket(
                task="unfinished research task",
                status="insufficient",
                summary="The worker did not finish within its step budget.",
                gaps=["Agent step budget exhausted"],
            ),
            graph_name="research-worker",
        )

    def _validate_fast_answer(self, answer: AgentAnswer) -> None:
        if self.workspace.search_count <= 0:
            raise ValueError("Search the knowledge base before submitting an answer")
        if not answer.evidence_ids:
            raise ValueError("A grounded answer requires at least one evidence ID")
        self.workspace.validate_evidence_ids(answer.evidence_ids)

    def _can_submit_fast_answer(self, args: dict[str, Any]) -> bool:
        try:
            self._validate_fast_answer(AgentAnswer.model_validate(args))
        except ValueError:
            return False
        return True

    def _delegate_tool(self) -> BaseTool:
        runtime = self

        @tool(args_schema=DelegateInput)
        def delegate_research(
            task: str, acceptance_criteria: list[str], config: RunnableConfig
        ) -> str:
            """Delegate one bounded research task to a ReAct worker."""

            with runtime._lock:
                if runtime._subtask_count >= runtime.max_subtasks:
                    raise ValueError("Research subtask budget exhausted")
                runtime._subtask_count += 1
            request = task + "\nAcceptance criteria:\n- " + "\n- ".join(
                acceptance_criteria
            )
            # A fresh worker graph gives every delegated task an isolated message
            # history and evidence-read budget while sharing the evidence registry.
            worker_graph = runtime._build_worker_graph()
            state = worker_graph.invoke(
                {"messages": [HumanMessage(content=request)], "steps": 0},
                {
                    **config,
                    "run_name": "research-worker",
                    "recursion_limit": runtime.max_steps * 2 + 4,
                },
            )
            packet = ResearchPacket.model_validate(state["result"])
            runtime._validate_packet(packet)
            with runtime._lock:
                runtime._packets.append(packet)
            return packet.model_dump_json()

        return delegate_research

    def _validate_packet(self, packet: ResearchPacket) -> None:
        cited = set(packet.evidence_ids)
        cited.update(
            citation.evidence_id
            for claim in packet.claims
            for citation in claim.citations
        )
        self.workspace.validate_evidence_ids(cited)
        evidence_by_id = self.workspace.evidence_by_id()
        packet.claims = [
            self.workspace.verifier.verify_claim(claim, evidence_by_id)
            for claim in packet.claims
        ]
        if packet.status == "sufficient" and (not packet.claims or not cited):
            raise ValueError(
                "A sufficient research packet requires verified claims and evidence"
            )

    def _validate_supervisor_answer(self, answer: AgentAnswer) -> None:
        sufficient = [item for item in self.packets if item.status == "sufficient"]
        allowed_ids = {
            evidence_id
            for packet in sufficient
            for evidence_id in packet.evidence_ids
        }
        allowed_ids.update(
            citation.evidence_id
            for packet in sufficient
            for claim in packet.claims
            for citation in claim.citations
        )
        if not sufficient or not allowed_ids:
            raise ValueError(
                "No sufficient worker packet with verified evidence is available"
            )
        if not answer.evidence_ids:
            raise ValueError("The final deliverable requires worker evidence IDs")
        unknown = set(answer.evidence_ids) - allowed_ids
        if unknown:
            raise ValueError(
                f"Final answer cites evidence not returned by workers: {sorted(unknown)}"
            )

    def _can_submit_supervisor_answer(self, args: dict[str, Any]) -> bool:
        try:
            self._validate_supervisor_answer(AgentAnswer.model_validate(args))
        except ValueError:
            return False
        return True

    def _build_supervisor_graph(self):
        tools = [
            self._delegate_tool(),
            self._submit_answer_tool(self._validate_supervisor_answer),
        ]
        return self._build_react_graph(
            prompt=SUPERVISOR_PROMPT,
            tools=tools,
            submit_name="submit_answer",
            result_model=AgentAnswer,
            exhausted_result=lambda: AgentAnswer(
                content="研究调度未能在执行预算内完成。",
                limitations=["Supervisor step budget exhausted"],
            ),
            can_submit=self._can_submit_supervisor_answer,
            graph_name="research-supervisor",
        )

    def _build_root_graph(self):
        router = self.model.with_structured_output(RouteDecision, method="json_mode")

        def route(state: RootState, config: RunnableConfig) -> dict:
            decision = router.invoke(
                [
                    SystemMessage(content=ROUTER_PROMPT),
                    HumanMessage(content=state["request"]),
                ],
                config=config,
            )
            return {"route": decision.model_dump(mode="json")}

        def route_mode(state: RootState) -> str:
            return state["route"]["mode"]

        def fast(state: RootState, config: RunnableConfig) -> dict:
            result = self._fast_graph.invoke(
                {"messages": [HumanMessage(content=state["request"])], "steps": 0},
                {
                    **config,
                    "run_name": "fast-react-agent",
                    "recursion_limit": self.max_steps * 2 + 4,
                },
            )
            return {"answer": result["result"]}

        def supervisor(state: RootState, config: RunnableConfig) -> dict:
            result = self._supervisor_graph.invoke(
                {"messages": [HumanMessage(content=state["request"])], "steps": 0},
                {
                    **config,
                    "run_name": "research-supervisor",
                    "recursion_limit": self.max_steps * 2 + 4,
                    "max_concurrency": self.max_workers,
                },
            )
            return {"answer": result["result"]}

        builder = StateGraph(RootState)
        builder.add_node("router", route)
        builder.add_node("fast_agent", fast)
        builder.add_node("supervisor_agent", supervisor)
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            route_mode,
            {"fast": "fast_agent", "supervisor": "supervisor_agent"},
        )
        builder.add_edge("fast_agent", END)
        builder.add_edge("supervisor_agent", END)
        return builder.compile(name="research-router")

    def run(
        self, request: str, *, config: RunnableConfig | None = None
    ) -> tuple[AgentRun, Any]:
        if not request.strip():
            raise ValueError("request must not be blank")
        with self._run_lock:
            self.workspace.reset()
            with self._lock:
                self._packets.clear()
                self._subtask_count = 0
            state = self.graph.invoke(
                {"request": request.strip()},
                config or {"recursion_limit": self.max_steps * 4 + 8},
            )
            route = RouteDecision.model_validate(state["route"])
            answer = AgentAnswer.model_validate(state["answer"])
            self.workspace.validate_evidence_ids(answer.evidence_ids)
            run = AgentRun(
                request=request.strip(),
                route=route,
                answer=answer,
                evidence=self.workspace.evidence,
                worker_packets=self.packets,
            )
            path = self.store.save(run)
            return run, path
