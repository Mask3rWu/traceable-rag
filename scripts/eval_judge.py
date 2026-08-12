"""LLM-as-judge packaging / scoring utility for runtime-evaluation batches.

The runtime evaluator (``scripts/eval_runtime.py``) reports *structural*
telemetry (route matching, cost, retrieval citation coverage, delivery
coverage) but never judges *content* quality. This tool supports an
LLM-as-judge pass over a finished batch:

- ``--case <qid>``  package a per-question, LLM-ready digest (question +
  expected rubric + answer + the *cited* evidence quotes, budget-capped).
  The biggest challenge is that supervisor runs carry 1.5-1.9 MB of raw
  output; the packaging isolates just the cited evidence and truncates it
  to a token budget so a judging agent does not have to read raw files.
- ``--api MODEL``  optional *headless* scoring: package every selected case
  then call an OpenAI-compatible LLM (e.g. DeepSeek) to fill the report.
  This is for CI / no-interactive-agent scenarios. The default flow calls
  NO LLM -- the invoking agent (Codex / Claude Code) is the judge.
- ``--report``     render ``judge/report.md`` from verdicts already stored
  in ``judge/report.json``.

Outputs live inside the selected batch under ``eval/runtime/batches/<id>/judge/``.

Usage::

    python scripts/eval_judge.py --batch b_20260811_1957_backfill          # package all cases
    python scripts/eval_judge.py --case q6                                  # package one case
    python scripts/eval_judge.py --api deepseek-chat --api-key ...          # headless scoring
    python scripts/eval_judge.py --report                                   # render report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_ROOT = PROJECT_ROOT / "eval" / "runtime"
DEFAULT_QUESTIONS = EVAL_ROOT / "questions.yaml"
CASE_SCHEMA = "judge-case-v1"
REPORT_SCHEMA = "judge-report-v1"
VALID_CATEGORIES = ("direct", "borrow", "unrelated")
REQUIRED_FIELDS = ("id", "category", "expected_route", "question", "expected")

# Budget split (overridable via --token-budget). Overhead is the prompt glue
# (instructions + question + expected); the rest is split ~45/45 between the
# answer and the cited-evidence slice.
OVERHEAD_TOKENS = 500
ANSWER_FRACTION = 0.45
EVIDENCE_FRACTION = 0.45  # remainder after overhead; 0.45+0.45 < 1 keeps slack
_TOK_PER_CHAR = 1.1  # CJK-mixed heuristic: how many tokens one char tends to cost


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must map to an object: {path}")
    return data


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
            raise SystemExit(f"Question {item.get('id')!r} missing fields: {missing}")
        if item["id"] in seen:
            raise SystemExit(f"Duplicate question id: {item['id']}")
        seen.add(item["id"])
        if item["category"] not in VALID_CATEGORIES:
            raise SystemExit(
                f"Question {item['id']} has invalid category {item['category']!r}"
            )
    return questions


def resolve_batch_dir(batch: str | Path | None) -> Path:
    """Resolve a ``--batch`` arg to its directory.

    ``None`` -> the lexicographically-latest ``b_*`` dir (zero-padded names
    sort chronologically, matching the ``eval_runtime.py`` naming scheme).
    """
    if batch is None:
        candidates = sorted(
            p for p in (EVAL_ROOT / "batches").glob("b_*") if p.is_dir()
        )
        if not candidates:
            raise SystemExit("no batches under eval/runtime/batches/")
        return candidates[-1]
    p = Path(batch)
    if p.exists():
        return p.resolve()
    d = (EVAL_ROOT / "batches" / str(batch)).resolve()
    if not d.is_dir():
        raise SystemExit(f"batch dir not found: {d}")
    return d


# -- token / truncation helpers -------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Heuristic token estimate without a tiktoken dependency.

    CJK chars dominate this corpus (~87%) and cost more than ASCII: ~1.2
    tokens/char for CJK, ~0.25 for ASCII. Deterministic and cheap.
    """
    return int(sum(1.2 if ord(c) > 127 else 0.25 for c in text))


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Deterministic head+tail splice with an explicit omission marker.

    Returns ``(text, truncated)``. Keeps the first ~60% and last ~40% of the
    budget with a ``… [N chars omitted] …`` marker between them, so the start
    and the end of a quote/answer both survive.
    """
    if len(text) <= max_chars:
        return text, False
    marker = f"\n… [省略 {len(text) - max_chars} 字符] …\n"
    budget = max_chars - len(marker)
    if budget <= 4:
        return text[: max(4, max_chars)], True
    head_len = int(budget * 0.6)
    return text[:head_len] + marker + text[-(budget - head_len):], True


def _pages(evidence: dict) -> int | str | None:
    start, end = evidence.get("page_start"), evidence.get("page_end")
    if start is None and end is None:
        return None
    if end is None or start == end:
        return start
    return f"{start}-{end}"


def _cited_ordered(evidence: list[dict], cited_ids: list[str]) -> list[dict]:
    """Cited evidence in answer citation order, deduped by (source, page)."""
    by_id = {e.get("evidence_id"): e for e in evidence}
    seen: set[tuple] = set()
    ordered: list[dict] = []
    for eid in cited_ids:
        e = by_id.get(eid)
        if not e:
            continue
        key = (e.get("source_file"), e.get("page_start"))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(e)
    return ordered


def select_and_truncate_evidence(
    evidence: list[dict], cited_ids: list[str], budget_tokens: int
) -> dict:
    """Stable, budget-aware packing of the *cited* evidence.

    Only evidence actually cited by the answer is kept (it is what survived
    synthesis); each quote is truncated to a per-quote cap, then quotes are
    greedily accumulated until the evidence token budget is spent. Anything
    beyond the cap is counted in ``omitted`` (never silently dropped).
    """
    ordered = _cited_ordered(evidence, cited_ids)
    per_quote_chars = max(120, int(budget_tokens * _TOK_PER_CHAR // 40))
    total_before = sum(len(e.get("quote") or e.get("quote_truncated") or "") for e in ordered)
    items: list[dict] = []
    used = 0
    for e in ordered:
        quote = e.get("quote") or e.get("quote_truncated") or ""
        q, truncated = truncate_text(quote, per_quote_chars)
        cost = estimate_tokens(q)
        if used + cost > budget_tokens:
            break
        used += cost
        items.append(
            {
                "evidence_id": e.get("evidence_id"),
                "source_file": e.get("source_file"),
                "pages": _pages(e),
                "section_path": e.get("section_path") or [],
                "quote": q,
                "quote_truncated": truncated,
            }
        )
    return {
        "items": items,
        "total_cited": len(ordered),
        "included": len(items),
        "omitted": len(ordered) - len(items),
        "omitted_reason": "token budget",
        "quote_chars_before": total_before,
        "quote_chars_after": sum(len(i["quote"]) for i in items),
    }


# -- per-question case packaging -------------------------------------------------

def _packet_statuses(run: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in run.get("worker_packets") or []:
        status = p.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def package_case(
    question: dict, run: dict, metrics: dict, *, token_budget: int
) -> dict:
    """Build one LLM-ready case dict from a question + full AgentRun + metrics.

    Pure and deterministic. The answer keeps ~45% of the budget, the cited
    evidence slice ~45%, so a judging agent can grade faithfulness (does the
    answer's content come from the cited quotes?) without reading the raw
    1.5-1.9 MB run files.
    """
    budget = {
        "token_budget": token_budget,
        "estimate_method": "CJK*1.2 + ASCII*0.25 heuristic",
        "overhead_tokens": OVERHEAD_TOKENS,
    }
    evidence_tokens = int((token_budget - OVERHEAD_TOKENS) * EVIDENCE_FRACTION)
    answer_tokens = token_budget - OVERHEAD_TOKENS - evidence_tokens
    answer_chars_cap = int(answer_tokens / _TOK_PER_CHAR)

    answer = run.get("answer") or {}
    content = answer.get("content") or ""
    content, content_truncated = truncate_text(content, answer_chars_cap)
    budget["answer_tokens"] = estimate_tokens(content)
    budget["answer_truncated"] = content_truncated

    cited_ids = answer.get("evidence_ids") or []
    evidence_pack = select_and_truncate_evidence(
        run.get("evidence") or [], cited_ids, evidence_tokens
    )
    budget["evidence_tokens"] = sum(
        estimate_tokens(item["quote"]) for item in evidence_pack["items"]
    )
    budget["evidence_included"] = evidence_pack["included"]
    budget["evidence_omitted"] = evidence_pack["omitted"]

    download = metrics or {}
    return {
        "schema_version": CASE_SCHEMA,
        "batch_id": Path(run.get("_batch_dir", "")).name if run.get("_batch_dir") else None,
        "question_id": question["id"],
        "budget": budget,
        "question": {
            "id": question["id"],
            "category": question["category"],
            "expected_route": question["expected_route"],
            "question": question["question"],
            "expected": question.get("expected", ""),
        },
        "run": {
            "outcome": run.get("outcome"),
            "error": run.get("error"),
            "route_matched": download.get("route_matched"),
            "route": (run.get("route") or {}).get("mode"),
            "elapsed_s": download.get("elapsed_s"),
            "model_calls_count": (download.get("model_calls") or {}).get("count"),
            "consistency_issues": run.get("consistency_issues") or [],
            "packet_statuses": _packet_statuses(run),
            "answer": {
                "content": content,
                "content_truncated": content_truncated,
                "limitations": answer.get("limitations") or [],
            },
            "evidence": evidence_pack["items"],
            "_evidence_meta": {
                k: v for k, v in evidence_pack.items() if k != "items"
            },
        },
    }


# -- report scaffolding -----------------------------------------------------------

def _case_run(batch_dir: Path, qid: str) -> dict | None:
    """Load the full AgentRun for a question, or None if only a checkpoint exists."""
    run_path = batch_dir / "runs" / f"{qid}.json"
    if not run_path.is_file():
        return None
    with open(run_path, encoding="utf-8") as fh:
        return json.load(fh)


def _case_metrics(batch_dir: Path, qid: str) -> dict:
    path = batch_dir / f"{qid}.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _build_cases(batch_dir: Path, questions: list[dict], qids: list[str],
                 token_budget: int, overwrite: bool) -> list[dict]:
    judge_dir = batch_dir / "judge"
    if judge_dir.is_dir() and any(judge_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"{judge_dir} already exists. Use --overwrite to regenerate."
        )
    cases: list[dict] = []
    for question in questions:
        if question["id"] not in qids:
            continue
        run = _case_run(batch_dir, question["id"])
        if run is None:
            cases.append(
                {
                    "question_id": question["id"],
                    "status": "skipped",
                    "reason": f"no runs/{question['id']}.json (checkpoint only)",
                }
            )
            continue
        metrics = _case_metrics(batch_dir, question["id"])
        run["_batch_dir"] = str(batch_dir)
        case = package_case(question, run, metrics, token_budget=token_budget)
        _write_json(judge_dir / f"case_{question['id']}.json", case)
        cases.append({"question_id": question["id"], "status": "pending"})
    return cases


def _write_report_scaffold(batch_dir: Path, cases: list[dict]) -> None:
    report_json = {
        "schema_version": REPORT_SCHEMA,
        "batch_id": batch_dir.name,
        "git_commit": _git_commit(),
        "created_at": _utc_iso(),
        "cases": cases,
        "aggregate": {"as_of": "pending"},
    }
    _write_json(batch_dir / "judge" / "report.json", report_json)

    pending = [c for c in cases if c["status"] == "pending"]
    lines = [
        f"# LLM-as-judge 评估报告 {batch_dir.name}",
        "",
        f"- schema_version: `{REPORT_SCHEMA}`",
        f"- git_commit: `{_git_commit()}`",
        f"- created_at: `{_utc_iso()}`",
        "",
        "## 汇总",
        "",
        "| 题 | 类 | 路由匹配 | 正确性 | 忠实度 | 完整性 | 一致性 | 诚实性 | 建议 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        lines.append(f"| {c['question_id']} | - | - | - | - | - | - | - | - |")
    lines += [
        "",
        "<!-- 等级分布与跨题模式 -->",
        "",
        "## 逐题判定",
        "",
    ]
    for c in cases:
        lines += [
            f"### {c['question_id']}",
            "",
            f"状态：`{c['status']}`"
            + (f"（{c.get('reason', '')}）" if c.get("reason") else ""),
            "",
        ]
        if c["status"] == "pending":
            lines += [
                "<!-- FILL：逐维 A/B/C/D + 理由 + 优缺点（表格） -->",
                "| 维度 | 等级 | 理由 | 优点 | 缺点 |",
                "|---|---|---|---|---|",
                "| 正确性 | `<!-- tier -->` | <!-- 理由 --> | <!-- 优点 --> | <!-- 缺点 --> |",
                "| 忠实度/无幻觉 | `<!-- tier -->` | <!-- 理由 --> | <!-- 优点 --> | <!-- 缺点 --> |",
                "| 完整性 | `<!-- tier -->` | <!-- 理由 --> | <!-- 优点 --> | <!-- 缺点 --> |",
                "| 一致性 | `<!-- tier -->` | <!-- 理由 --> | <!-- 优点 --> | <!-- 缺点 --> |",
                "| 诚实性 | `<!-- tier -->` | <!-- 理由 --> | <!-- 优点 --> | <!-- 缺点 --> |",
                "",
                "- 总体优点：",
                "- 总体缺点：",
                "- 结论 / 建议（approve|revise|reject）：",
                "",
            ]
    (batch_dir / "judge" / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# -- optional headless --api scoring -------------------------------------------------

JUDGE_RUBRIC = """\
你是舆情/知识库 RAG 研究 Agent 输出的内容质量判官。针对给定问题、其类别与期望(
expected)、以及被引证据引文，对 Agent 的最终答案逐维评级。只依据提供的 case 内容
判断，不重跑、不依据外部知识做编造。

评价维度（每维给 A/B/C/D 四级 + 一条理由 + 具体优点/缺点示例）：
- 正确性：事实是否正确；direct=与源一致，borrow=推断合理，unrelated=通用知识正确。
- 忠实度/无幻觉：每条实质主张是否被引用证据引文支撑；有无虚构/编造引用、有无超出
  证据的断言。
- 完整性：是否覆盖问题全部所指；supervisor=是否覆盖请求的交付面。
- 一致性：答案内部/跨章节是否有自相矛盾（可综合 case 中 consistency_issues）。
- 诚实性：unrelated 是否诚实声明无资料而不虚构；borrow 是否标注 hypothesis/设计值。

supervisor 题额外：
- 可操作性：规则能否落地（阈值已定义、无未定义组合、覆盖目标场景）。

输出严格为 JSON（不要 markdown 围栏），形如：
{
  "question_id": "...",
  "dimensions": [{"name":"正确性","tier":"A|B|C|D","reason":"...","pros":["..."],"cons":["..."]}],
  "overall_strengths":["..."],
  "overall_weaknesses":["..."],
  "conclusion":"...",
  "recommendation":"approve|revise|reject"
}
注意：tier 只是附注，理由与优缺点才是重点，须具体、引用原文/证据佐证。
"""


def _judge_one(case: dict, model: str, base_url: str, api_key: str) -> dict:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "--api requires langchain-openai (pip install langchain-openai): " + str(exc)
        )
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)
    prompt = (
        JUDGE_RUBRIC
        + "\n\n以下为待判 case（JSON）：\n"
        + json.dumps(case, ensure_ascii=False, indent=2)
    )
    response = llm.invoke(prompt)
    text = response.content if isinstance(response, object) else str(response)
    if isinstance(text, list):
        text = "".join(str(part.get("text", part)) for part in text)
    match = re.search(r"\{.*\}", str(text), re.S)
    if not match:
        raise RuntimeError(f"judge did not return JSON: {str(text)[:300]}")
    return json.loads(match.group(0))


def _score_with_api(batch_dir: Path, cases: list[dict], model: str,
                    base_url: str, api_key: str) -> list[dict]:
    judge_progress = []
    for c in cases:
        if c["status"] != "pending":
            continue
        qid = c["question_id"]
        case = json.loads(
            (batch_dir / "judge" / f"case_{qid}.json").read_text(encoding="utf-8")
        )
        verdict = _judge_one(case, model, base_url, api_key)
        c["verdict"] = verdict
        judge_progress.append(qid)
        print(f"  judged {qid}")
    return judge_progress


def _tier(dimensions: list[dict], name: str) -> str:
    for d in dimensions:
        if d.get("name") == name:
            return d.get("tier", "-")
    return "-"


def render_report(batch_dir: Path, report_json: dict) -> None:
    cases = report_json.get("cases", [])

    def row(c: dict) -> str:
        verdict = c.get("verdict")
        if not verdict:
            return f"| {c.get('question_id')} | - | - | - | - | - | - | - |"
        dims = verdict.get("dimensions", [])
        return (
            f"| {c.get('question_id')} | - | {_tier(dims,'正确性')} | "
            f"{_tier(dims,'忠实度/无幻觉')} | {_tier(dims,'完整性')} | "
            f"{_tier(dims,'一致性')} | {_tier(dims,'诚实性')} | "
            f"{verdict.get('recommendation')} |"
        )

    lines = [
        f"# LLM-as-judge 评估报告 {batch_dir.name}",
        "",
        f"- schema_version: `{report_json.get('schema_version')}`",
        f"- git_commit: `{report_json.get('git_commit')}`",
        f"- created_at: `{report_json.get('created_at')}`",
        "",
        "## 汇总",
        "",
        "| 题 | 类 | 正确性 | 忠实度 | 完整性 | 一致性 | 诚实性 | 建议 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [row(c) for c in cases]
    lines += ["", "<!-- 等级分布与跨题模式 -->", "", "## 逐题判定", ""]
    for c in cases:
        qid = c.get("question_id")
        lines += [f"### {qid}", "", f"状态：`{c.get('status')}`", ""]
        verdict = c.get("verdict")
        if not verdict:
            lines += ["<!-- 未判定 -->", ""]
            continue
        dims = verdict.get("dimensions", [])
        if dims:
            lines += [
                "| 维度 | 等级 | 理由 | 优点 | 缺点 |",
                "|---|---|---|---|---|",
            ]
            for dim in dims:
                lines.append(
                    f"| {dim.get('name')} | `{dim.get('tier')}` | "
                    f"{dim.get('reason')} | "
                    f"{'<br>'.join(dim.get('pros') or [])} | "
                    f"{'<br>'.join(dim.get('cons') or [])} |"
                )
            lines += [""]
        lines += ["- 优点："]
        lines += [f"  - {s}" for s in verdict.get("overall_strengths", [])]
        lines += ["- 缺点："]
        lines += [f"  - {w}" for w in verdict.get("overall_weaknesses", [])]
        lines += ["", f"- 结论：{verdict.get('conclusion')}"]
        lines += [f"- 建议：`{verdict.get('recommendation')}`", ""]
    (batch_dir / "judge" / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# -- CLI ---------------------------------------------------------------------------

def _parse_qids(limit: str | None, questions: list[dict]) -> list[str]:
    if not limit:
        return [q["id"] for q in questions]
    if limit.isdigit():
        return [q["id"] for q in questions[: int(limit)]]
    return [s.strip() for s in limit.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge packaging / scoring for runtime-eval batches."
    )
    parser.add_argument("--batch", default=None,
                        help="batch id (b_...) or path; default = latest")
    parser.add_argument("--case", default=None,
                        help="comma-separated qids or integer N to package")
    parser.add_argument("--token-budget", type=int, default=8000,
                        help="per-case token budget (default 8000; raise to 10000 for near-full answers)")
    parser.add_argument("--out", default=None, help="override output dir (default <batch>/judge)")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate an existing judge/ dir")
    parser.add_argument("--report", action="store_true",
                        help="render judge/report.md from existing report.json verdicts")
    parser.add_argument("--api", default=None, metavar="MODEL",
                        help="optional headless scoring: call this OpenAI-compatible model")
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key", default=None, help="API key (or env DEEPSEEK_API_KEY)")
    args = parser.parse_args(argv)

    batch_dir = resolve_batch_dir(args.batch)
    questions = _validate_questions(_load_yaml(DEFAULT_QUESTIONS))
    qids = _parse_qids(args.case, questions)

    if args.report:
        report_path = batch_dir / "judge" / "report.json"
        if not report_path.is_file():
            raise SystemExit(f"no {report_path} to render")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        render_report(batch_dir, report)
        print(f"rendered {batch_dir / 'judge' / 'report.md'}")
        return 0

    cases = _build_cases(batch_dir, questions, qids, args.token_budget, args.overwrite)

    if args.api:
        api_key = args.api_key or __import__("os").environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("--api requires --api-key or DEEPSEEK_API_KEY env")
        print(f"scoring {len([c for c in cases if c['status']=='pending'])} cases with {args.api} ...")
        _score_with_api(batch_dir, cases, args.api, args.api_base_url, api_key)
        report = {
            "schema_version": REPORT_SCHEMA,
            "batch_id": batch_dir.name,
            "git_commit": _git_commit(),
            "created_at": _utc_iso(),
            "cases": cases,
            "aggregate": {"as_of": _utc_iso()},
        }
        _write_json(batch_dir / "judge" / "report.json", report)
        render_report(batch_dir, report)
        print(f"wrote {batch_dir / 'judge' / 'report.json'}")
        print(f"wrote {batch_dir / 'judge' / 'report.md'}")
        return 0

    _write_report_scaffold(batch_dir, cases)
    print(f"batch: {batch_dir.name}")
    print(f"judged cases packaged under {batch_dir / 'judge'}")
    for c in cases:
        print(f"  - {c['question_id']}: {c['status']}")
    print(f"report scaffold: {batch_dir / 'judge' / 'report.md'}")
    print(
        "首次交互：由调用方 agent（Codex/Claude Code）逐个读 case 并按 rubric 判定，"
        "填 report.md；或加 --api <model> 走无头自动评分。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())