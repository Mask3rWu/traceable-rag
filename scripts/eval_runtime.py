"""Run the runtime-evaluation batch by driving the research API.

Each evaluation question is submitted through ``POST /api/runs`` exactly as if
typed into the web workbench, so every run appears in the normal run list with
real-time SSE progress. This means the web UI is the live view; this runner's
job is to (a) submit the questions, (b) wait for each to reach a terminal
state, (c) back up the full output, and (d) produce a metrics report.

Every batch is versioned and isolated under ``eval/runtime/batches/``; the
backed-up outputs there are independent of the run store (which may be cleared
or restructured), so historical eval data survives database resets.

Fast tasks carry a route policy (``expected_route: fast``): if the router
decides to run such a task as a multi-agent (supervisor) task, the API
interrupts it before the expensive subgraph (status=``routed_away``) and the
report records the router's mode + reason so you can see *why* it happened.

Usage::

    python scripts/eval_runtime.py                      # full batch via API
    python scripts/eval_runtime.py --dry-run             # validate request set
    python scripts/eval_runtime.py --phase fast          # only fast tasks
    python scripts/eval_runtime.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.eval_metrics import RuntimeMetrics  # noqa: E402

EVAL_ROOT = PROJECT_ROOT / "eval" / "runtime"
DEFAULT_QUESTIONS = EVAL_ROOT / "questions.yaml"
DEFAULT_PRICING = EVAL_ROOT / "pricing.yaml"
SCHEMA_VERSION = "runtime-eval-v2"
REQUIRED_FIELDS = ("id", "category", "expected_route", "question", "expected")
VALID_CATEGORIES = ("direct", "borrow", "unrelated")
TERMINAL_STATUSES = {"completed", "incomplete", "failed", "cancelled", "routed_away"}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must map to an object: {path}")
    return data


def _git_commit() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _validate_questions(data: dict) -> list[dict]:
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise SystemExit("questions.yaml must contain a non-empty 'questions' list")
    seen: set[str] = set()
    for item in questions:
        if not isinstance(item, dict):
            raise SystemExit("Each question must be a mapping")
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            raise SystemExit(
                f"Question {item.get('id')!r} missing fields: {missing}"
            )
        if item["id"] in seen:
            raise SystemExit(f"Duplicate question id: {item['id']}")
        seen.add(item["id"])
        if item["category"] not in VALID_CATEGORIES:
            raise SystemExit(
                f"Question {item['id']} has invalid category {item['category']!r}"
            )
    return questions


def _question_set_hash(questions: list[dict]) -> str:
    canonical = json.dumps(questions, ensure_ascii=False, sort_keys=True)
    return _sha256_text(canonical)


def _category_counts(questions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in questions:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    return counts


# -- API client --------------------------------------------------------------


def _preflight(base_url: str) -> None:
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"Research API not reachable at {base_url}: {exc}. "
            "Start it first (scripts/run_api.py)."
        ) from exc


def submit_run(base_url: str, question: dict) -> str:
    payload = {"request": question["question"]}
    if question.get("expected_route"):
        payload["expected_route"] = question["expected_route"]
    response = requests.post(f"{base_url}/api/runs", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["run_id"]


def get_run(base_url: str, run_id: str) -> dict:
    response = requests.get(f"{base_url}/api/runs/{run_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def poll_terminal(
    base_url: str,
    run_id: str,
    poll_interval: float,
    timeout: float = 4 * 3600,
) -> tuple[dict, float]:
    started = time.perf_counter()
    while True:
        detail = get_run(base_url, run_id)
        if detail.get("status") in TERMINAL_STATUSES:
            return detail, time.perf_counter() - started
        if time.perf_counter() - started > timeout:
            raise TimeoutError(
                f"run {run_id} did not reach a terminal state within {timeout}s"
            )
        time.sleep(poll_interval)


# -- metrics / report --------------------------------------------------------


def compute_retrieval(result: dict | None, metrics: RuntimeMetrics) -> dict:
    """Derive retrieval metrics from the API-returned AgentRun + metrics."""
    if result is None:
        return {
            "search_count": metrics.search_count(),
            "search_latency_ms": metrics.search_latencies_ms(),
            "evidence_total": 0,
            "evidence_unique": 0,
            "evidence_cited": 0,
            "dedup_ratio": None,
            "citation_coverage": None,
        }
    evidence_cited: set[str] = set((result.get("answer") or {}).get("evidence_ids") or [])
    for packet in result.get("worker_packets") or []:
        evidence_cited.update(packet.get("evidence_ids") or [])
    evidence = result.get("evidence") or []
    evidence_total = sum(len(item.get("retrieval") or []) for item in evidence)
    evidence_unique = len(evidence)
    return {
        "search_count": metrics.search_count(),
        "search_latency_ms": metrics.search_latencies_ms(),
        "evidence_total": evidence_total,
        "evidence_unique": evidence_unique,
        "evidence_cited": len(evidence_cited),
        "dedup_ratio": round(evidence_unique / evidence_total, 4)
        if evidence_total
        else None,
        "citation_coverage": round(len(evidence_cited) / evidence_unique, 4)
        if evidence_unique
        else None,
    }


def compute_delivery(result: dict | None) -> dict:
    """Derive deliverable-coverage counters from the persisted AgentRun.

    Purely structural: no content judgement and no reliance on model-authored
    audit fields. Counts the plan/execution/assembly coverage so the numbers
    survive claim/decision/evidence field refactors.
    """
    if result is None:
        return {
            "planned_chapters": None,
            "executed_packets": 0,
            "packet_sufficient": 0,
            "packet_insufficient": 0,
            "packet_failed": 0,
            "packet_blocked": 0,
            "assembled": False,
            "answer_chars": 0,
            "answer_evidence": 0,
        }
    chapters = (result.get("document_plan") or {}).get("chapters") or []
    packets = result.get("worker_packets") or []
    statuses: dict[str, int] = {}
    for packet in packets:
        status = packet.get("status") or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    answer = result.get("answer") or {}
    return {
        "planned_chapters": len(chapters) or None,
        "executed_packets": len(packets),
        "packet_sufficient": statuses.get("sufficient", 0),
        "packet_insufficient": statuses.get("insufficient", 0),
        "packet_failed": statuses.get("failed", 0),
        "packet_blocked": statuses.get("blocked", 0),
        "assembled": bool(answer.get("content")),
        "answer_chars": len(answer.get("content") or ""),
        "answer_evidence": len(answer.get("evidence_ids") or []),
    }


def run_one(
    base_url: str,
    question: dict,
    pricing: dict,
    batch_dir: Path,
    sequence: int,
    poll_interval: float,
) -> dict:
    qid = question["id"]
    run_id = submit_run(base_url, question)
    detail, elapsed = poll_terminal(base_url, run_id, poll_interval)
    result = detail.get("result")
    metrics = RuntimeMetrics.from_snapshot(detail.get("metrics") or {})
    model_calls = metrics.model_calls_summary(pricing)
    model_calls["schema_validation"] = metrics.schema_validation_summary()
    retrieval = compute_retrieval(result, metrics)
    if result is not None:
        route = {
            "mode": result["route"]["mode"],
            "reason": result["route"]["reason"],
        }
    else:
        route = {"mode": detail.get("route"), "reason": detail.get("route_reason")}
    expected = question.get("expected_route")
    route_matched = route["mode"] == expected if route["mode"] is not None else None
    tool_calls = metrics.tool_calls_summary()
    schema = model_calls["schema_validation"]
    schema_failures = schema.get("failures") or []
    item = {
        "question": {
            "id": qid,
            "sequence": sequence,
            "category": question["category"],
            "expected_route": question.get("expected_route"),
        },
        "run_id": run_id,
        "outcome": detail.get("status"),
        "error": detail.get("error"),
        "ended_at": detail.get("updated_at"),
        "route": route,
        "route_matched": route_matched,
        "retrieval": retrieval,
        "delivery": compute_delivery(result),
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "steps_retries": {
            "model_calls": model_calls["count"],
            "tool_calls": tool_calls["count"],
            "submit_retries": sum(
                1 for failure in schema_failures if failure["stage"].startswith("submit")
            ),
            "plan_retries": sum(
                1 for failure in schema_failures if failure["stage"] == "plan"
            ),
        },
        "phase_cost": metrics.phase_summary(pricing),
        "elapsed_s": round(elapsed, 2),
    }
    _write_json(batch_dir / f"{qid}.json", item)
    if result is not None:
        backup = batch_dir / "runs"
        backup.mkdir(parents=True, exist_ok=True)
        _write_json(backup / f"{qid}.json", result)
    print(
        f"  #{sequence} {qid} -> {item['outcome']} ({item['elapsed_s']}s, "
        f"{item['model_calls']['count']} llm)"
    )
    return item


def _gather_tool_stats(results: list[dict]) -> dict:
    count = 0
    success = 0
    by_tool: dict[str, dict] = {}
    failures: list[dict] = []
    for result in results:
        tool_calls = result["tool_calls"]
        count += tool_calls["count"]
        failures.extend(tool_calls["failures"])
        for name, bucket in tool_calls["by_tool"].items():
            target = by_tool.setdefault(
                name, {"count": 0, "success": 0, "failed": 0, "total_latency_ms": 0}
            )
            target["count"] += bucket["count"]
            target["success"] += bucket["success"]
            target["failed"] += bucket["failed"]
            target["total_latency_ms"] += bucket["total_latency_ms"]
        success += sum(bucket["success"] for bucket in tool_calls["by_tool"].values())
    return {
        "count": count,
        "success_rate": round(success / count, 4) if count else None,
        "by_tool": by_tool,
        "failures": failures[:50],
    }


def _gather_summary(results: list[dict]) -> dict:
    model_count = 0
    schema_attempts = schema_passed = 0
    model_failures: list[dict] = []
    schema_failures: list[dict] = []
    retrieval = {
        "search_count": 0,
        "evidence_total": 0,
        "evidence_unique": 0,
        "evidence_cited": 0,
    }
    costs = []
    phases: dict[str, dict] = {}
    delivery: dict[str, int] = {
        "questions": 0,
        "planned_chapters": 0,
        "executed_packets": 0,
        "packet_sufficient": 0,
        "packet_insufficient": 0,
        "packet_failed": 0,
        "packet_blocked": 0,
        "assembled": 0,
        "answer_chars": 0,
        "answer_evidence": 0,
    }
    for result in results:
        model_calls = result["model_calls"]
        model_count += model_calls["count"]
        if model_calls["total_cost_usd"] is not None:
            costs.append(model_calls["total_cost_usd"])
        model_failures.extend(model_calls["failures"])
        schema = model_calls["schema_validation"]
        schema_attempts += schema["attempts"]
        schema_passed += schema["passed"]
        schema_failures.extend(schema["failures"])
        r = result["retrieval"]
        retrieval["search_count"] += r["search_count"]
        retrieval["evidence_total"] += r["evidence_total"]
        retrieval["evidence_unique"] += r["evidence_unique"]
        retrieval["evidence_cited"] += r["evidence_cited"]
        for phase, stats in (result.get("phase_cost") or {}).items():
            bucket = phases.setdefault(
                phase,
                {
                    "questions": 0,
                    "model_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "has_cost": True,
                    "total_latency_ms": 0.0,
                    "failed": 0,
                },
            )
            bucket["questions"] += 1
            bucket["model_calls"] += stats["model_calls"]
            bucket["prompt_tokens"] += stats["prompt_tokens"]
            bucket["completion_tokens"] += stats["completion_tokens"]
            bucket["total_latency_ms"] += stats["total_latency_ms"]
            bucket["failed"] += stats["failed"]
            if stats["cost_usd"] is None:
                bucket["has_cost"] = False
            else:
                bucket["cost_usd"] += stats["cost_usd"]
        d = result["delivery"]
        delivery["questions"] += 1
        delivery["planned_chapters"] += d["planned_chapters"] or 0
        delivery["executed_packets"] += d["executed_packets"]
        delivery["packet_sufficient"] += d["packet_sufficient"]
        delivery["packet_insufficient"] += d["packet_insufficient"]
        delivery["packet_failed"] += d["packet_failed"]
        delivery["packet_blocked"] += d["packet_blocked"]
        delivery["assembled"] += 1 if d["assembled"] else 0
        delivery["answer_chars"] += d["answer_chars"]
        delivery["answer_evidence"] += d["answer_evidence"]
    phase_cost = {
        phase: {
            "questions": bucket["questions"],
            "model_calls": bucket["model_calls"],
            "prompt_tokens": bucket["prompt_tokens"],
            "completion_tokens": bucket["completion_tokens"],
            "cost_usd": round(bucket["cost_usd"], 6) if bucket["has_cost"] else None,
            "total_latency_ms": round(bucket["total_latency_ms"], 1),
            "failed": bucket["failed"],
        }
        for phase, bucket in phases.items()
    }
    return {
        "model_calls": {
            "count": model_count,
            "total_cost_usd": round(sum(costs), 6) if costs else None,
            "schema_validation": {
                "attempts": schema_attempts,
                "passed": schema_passed,
                "failed": schema_attempts - schema_passed,
                "failures": schema_failures[:50],
            },
            "failures": model_failures[:50],
        },
        "tool_calls": _gather_tool_stats(results),
        "retrieval": retrieval,
        "delivery": delivery,
        "phase_cost": phase_cost,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown(batch_dir: Path, summary: dict, results: list[dict]) -> None:
    lines = [
        f"# 运行效果评估批次 {summary['batch_id']}",
        "",
        f"- schema_version: `{summary['schema_version']}`",
        f"- git_commit: `{summary['git_commit']}`",
        f"- created_at: {summary['created_at']}",
        f"- 问题数: {summary['question_set']['count']}",
        "",
        "## 逐题结果",
        "",
        "| 题 | 类 | 路由匹配 | outcome | 耗时(s) | 模型次数 | 成本($) | 工具次数 | 检索次数 | 去重引用 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item["question"]["sequence"]):
        q = result["question"]
        mc = result["model_calls"]
        tc = result["tool_calls"]
        r = result["retrieval"]
        cost = f"{mc['total_cost_usd']:.4f}" if mc["total_cost_usd"] is not None else "—"
        route = result.get("route") or {}
        expected = q.get("expected_route") or "—"
        actual = route.get("mode") or "—"
        matched = (
            "✓" if result.get("route_matched") else "✗" if result["route_matched"] is False else "—"
        )
        lines.append(
            f"| #{q['sequence']} {q['id']} | {q['category']} | "
            f"{expected}/{actual} {matched} | {result['outcome']} "
            f"| {result['elapsed_s']} | {mc['count']} | {cost} | {tc['count']} "
            f"| {r['search_count']} | {r['evidence_cited']} |"
        )
    agg = summary["aggregate"]
    by_outcome = summary.get("by_outcome") or {}
    lines += [
        "",
        "## 汇总",
        "",
        "- 状态分布: " + ", ".join(
            f"{label} {by_outcome.get(status, 0)}"
            for status, label in (
                ("completed", "completed"),
                ("incomplete", "incomplete"),
                ("failed", "failed"),
                ("routed_away", "routed_away"),
                ("cancelled", "cancelled"),
            )
        ),
        f"- 模型调用: {agg['model_calls']['count']} 次，成本 "
        + (
            f"${agg['model_calls']['total_cost_usd']:.4f}"
            if agg["model_calls"]["total_cost_usd"] is not None
            else "—"
        )
        + f"；schema 校验 {agg['model_calls']['schema_validation']['attempts']} 次，"
        f"通过 {agg['model_calls']['schema_validation']['passed']}，"
        f"失败 {agg['model_calls']['schema_validation']['failed']}",
        f"- 工具调用: {agg['tool_calls']['count']} 次，成功率 "
        + (
            f"{agg['tool_calls']['success_rate'] * 100:.1f}%"
            if agg["tool_calls"]["success_rate"] is not None
            else "—"
        ),
        f"- 检索: {agg['retrieval']['search_count']} 次，"
        f"去重前 {agg['retrieval']['evidence_total']}，去重后 {agg['retrieval']['evidence_unique']}，"
        f"实际引用 {agg['retrieval']['evidence_cited']}",
    ]
    delivery = agg["delivery"]
    if delivery["questions"]:
        lines.append(
            f"- 交付覆盖: 规划章节 {delivery['planned_chapters']}，执行 packet "
            f"{delivery['executed_packets']}（sufficient {delivery['packet_sufficient']} / "
            f"insufficient {delivery['packet_insufficient']} / failed "
            f"{delivery['packet_failed']} / blocked {delivery['packet_blocked']}），"
            f"已组装 {delivery['assembled']}/{delivery['questions']} 题"
        )
    routed = [
        r for r in results if r["outcome"] == "routed_away"
    ]
    if routed:
        lines += ["", "## 路由守卫中断（误判时立即中断）"]
        for result in routed:
            route = result.get("route") or {}
            lines.append(
                f"- #{result['question']['sequence']} {result['question']['id']}: "
                f"router 决策 mode={route.get('mode')!r}，reason="
                f"{route.get('reason')!r}"
            )
    failed = [r for r in results if r["outcome"] == "failed"]
    if failed:
        lines += ["", "## 失败明细"]
        for result in failed:
            lines.append(
                f"- #{result['question']['sequence']} {result['question']['id']}: "
                f"{(result.get('error') or 'no error captured')} "
                f"（{result['elapsed_s']}s）"
            )
    phases = agg["phase_cost"]
    if phases:
        lines += ["", "## 阶段成本归因（全批次汇总）", ""]
        lines.append(
            "| 阶段 | 参与题数 | 模型调用 | prompt(tok) | completion(tok) | 成本($) | 总耗时(s) | 失败 |"
        )
        lines.append(
            "|---|---|---:|---:|---:|---:|---:|---:|"
        )
        for phase, stats in phases.items():
            cost = f"{stats['cost_usd']:.4f}" if stats["cost_usd"] is not None else "—"
            lines.append(
                f"| {phase} | {stats['questions']} | {stats['model_calls']} "
                f"| {stats['prompt_tokens']} | {stats['completion_tokens']} "
                f"| {cost} | {stats['total_latency_ms']} | {stats['failed']} |"
            )
    (batch_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the runtime-evaluation batch via the API")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--batch-dir", type=Path, help="Override batch output dir")
    parser.add_argument("--concurrency", type=int, help="Override concurrency")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Research API base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3.0,
        help="Seconds between status polls while waiting for a run (default: 3)",
    )
    parser.add_argument(
        "--only",
        help="Run only a comma-separated subset of question ids (e.g. q1,q2)",
    )
    parser.add_argument(
        "--phase",
        choices=("fast", "supervisor", "all"),
        default="all",
        help="Limit to tasks expected to run fast / supervisor, or all (default)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    questions_data = _load_yaml(args.questions)
    pricing = _load_yaml(args.pricing)
    questions = _validate_questions(questions_data)
    if args.only:
        allowed = {item.strip() for item in args.only.split(",") if item.strip()}
        questions = [item for item in questions if item["id"] in allowed]
        if not questions:
            raise SystemExit(f"--only matched no questions: {args.only}")
    if args.phase != "all":
        questions = [
            item for item in questions if item.get("expected_route") == args.phase
        ]
        if not questions:
            raise SystemExit(f"--phase {args.phase} matched no questions")
    concurrency = args.concurrency or int(questions_data.get("concurrency", 1))
    if concurrency <= 0:
        raise SystemExit("--concurrency must be greater than zero")

    if args.dry_run:
        print(f"Request set: {args.questions}")
        print(f"  questions: {len(questions)}")
        print(f"  categories: {_category_counts(questions)}")
        print(f"  set hash: {_question_set_hash(questions)}")
        print(f"  concurrency: {concurrency}")
        for item in questions:
            print(f"    - {item['id']} [{item['category']}] {item['question'][:48]}")
        print("DRY RUN OK")
        return 0

    _preflight(args.base_url)

    batch_id = "b_" + datetime.now().strftime("%Y%m%d_%H%M")
    batch_dir = (args.batch_dir or (EVAL_ROOT / "batches" / batch_id)).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "batch_id": batch_id,
        "schema_version": SCHEMA_VERSION,
        "system_version": f"routed-research-agent@{_git_commit()}",
        "git_commit": _git_commit(),
        "created_at": _utc_iso(),
        "config_snapshot": {
            "backend": "api",
            "base_url": args.base_url,
            "concurrency": concurrency,
            "poll_interval_s": args.poll_interval,
            "pricing_models": sorted(pricing.keys()),
            "concurrency_requests": questions_data.get("concurrency"),
        },
        "question_set": {
            "file": str(args.questions),
            "hash": _question_set_hash(questions),
            "count": len(questions),
            "categories": _category_counts(questions),
        },
    }
    _write_json(batch_dir / "manifest.json", manifest)
    print(f"batch: {batch_id} -> {batch_dir}")
    print(f"submitting {len(questions)} questions to {args.base_url} (concurrency {concurrency})")

    results: list[dict] = []
    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="runtime-eval"
    ) as executor:
        futures = {
            executor.submit(
                run_one,
                args.base_url,
                question,
                pricing,
                batch_dir,
                sequence,
                args.poll_interval,
            ): question
            for sequence, question in enumerate(questions, start=1)
        }
        for future in as_completed(futures):
            question = futures[future]
            results.append(future.result())

    summary = {
        "batch_id": batch_id,
        "schema_version": SCHEMA_VERSION,
        "git_commit": manifest["git_commit"],
        "created_at": _utc_iso(),
        "question_set": manifest["question_set"],
        "total": len(results),
        "by_outcome": {
            status: sum(1 for item in results if item["outcome"] == status)
            for status in ("completed", "incomplete", "failed", "cancelled", "routed_away")
        },
        "aggregate": _gather_summary(results),
    }
    _write_json(batch_dir / "summary.json", summary)
    _write_markdown(batch_dir, summary, results)
    print(f"\nwrote summary: {batch_dir / 'summary.json'}")
    print(f"wrote report : {batch_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())