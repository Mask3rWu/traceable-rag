"""第一步：真实 agent 问题下的三家召回（dense / bm25 / RRF）。

背景：当前 schema 不落盘 agent 的中间检索词，只存"真实问题 request + 该问题
实际用过的 evidence chunk"。本工具用这两者做 Tier-R（相对比较开口、绝对
召回有自指上限）：
    - 单位 = 一个真实问题 request；
    - 相关集 R = 该 run 的 evidence chunk_ids（agent 真正用过的，视为标准答案）；
    - 对 request 分别用 dense/bm25/rrf 重跑，看谁把 R 排进 top-k。

只读分析，不改生产逻辑。用法：
    python scripts/diag_agent_evidence_recall.py [--candidate-limit 50]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROJECT_ROOT as ROOT  # noqa: E402
from src.retrieval.bm25 import BM25Retriever, DEFAULT_INDEX_ROOT  # noqa: E402
from src.retrieval.catalog import ChunkCatalog  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402

PATTERNS = [
    ROOT / "processed" / "research" / "eval-runs" / "*" / "run.json",
    ROOT / "eval" / "runtime" / "batches" / "**" / "runs" / "*.json",
]


def harvest() -> list[tuple[str, list[str]]]:
    """去重的问题 request + 可解 evidence chunk_ids（仅 completed）。"""
    by_request: dict[str, list[str]] = {}
    seen_runs = set()
    for pat in PATTERNS:
        for f in glob.glob(str(pat), recursive=True):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            rid = d.get("run_id")
            if rid and rid in seen_runs:
                continue
            if rid:
                seen_runs.add(rid)
            if d.get("outcome") != "completed":
                continue
            req = (d.get("request") or "").strip()
            if not req:
                continue
            eids = {e["chunk_id"] for e in (d.get("evidence") or [])}
            by_request.setdefault(req, set()).update(eids)
    return [(req, sorted(ev)) for req, ev in by_request.items() if ev]


def recall_at(top_k: list[str], relevant: set[str]) -> float:
    return len(set(top_k) & relevant) / len(relevant) if relevant else 0.0


def mrr(top_k: list[str], relevant: set[str]) -> float:
    for i, c in enumerate(top_k, start=1):
        if c in relevant:
            return 1.0 / i
    return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--ks", nargs="+", type=int, default=(5, 10, 20, 50))
    args = parser.parse_args(argv)

    catalog = ChunkCatalog.load()
    chunk_ids = set(catalog._sources)
    pairs = [(q, [c for c in ev if c in chunk_ids]) for q, ev in harvest()]
    pairs = [(q, ev) for q, ev in pairs if ev]
    if not pairs:
        raise SystemExit("no usable completed runs")
    print(f"真实问题数：{len(pairs)}")

    queries = [q for q, _ in pairs]
    ev = [set(e) for _, e in pairs]
    t0 = time.perf_counter()
    dense = DenseRetriever().search_many(queries, limit=args.candidate_limit)
    bm25 = BM25Retriever(DEFAULT_INDEX_ROOT).search_many(
        queries, limit=args.candidate_limit
    )
    rrf = [
        reciprocal_rank_fusion(di, bi, limit=args.candidate_limit)
        for di, bi in zip(dense, bm25, strict=True)
    ]
    print(f"检索耗时 {time.perf_counter()-t0:.1f}s")

    ranks: dict[str, list[str]] = {
        "dense": [[r.chunk_id for r in row] for row in dense],
        "bm25": [[r.chunk_id for r in row] for row in bm25],
        "rrf": [[r.chunk_id for r in row] for row in rrf],
    }

    print("\n=== 逐问题（真实问题 | evidence数 | 三家 top-10 命中/ED）===")
    for i, (q, rel) in enumerate(zip(queries, ev)):
        row = "  ".join(
            f"{m}:{len(set(ranks[m][i][:10]) & rel)}/{len(rel)}" for m in ranks
        )
        r = len(rel)
        print(f"  [{len(rel):>3} ev] {row} | {q[:58]}")

    print("\n=== 聚合均值（n=%d）===" % len(pairs))
    print(f"{'method':<6} " + "  ".join(f"R@{k:<4}" for k in args.ks) + "  MRR@10")
    agg: dict[str, dict] = {}
    for m in ranks:
        a = {}
        for k in args.ks:
            a[f"r@{k}"] = statistics.mean(
                recall_at(ranks[m][i][:k], rel) for i, rel in enumerate(ev)
            )
        a["mrr@10"] = statistics.mean(
            mrr(ranks[m][i][:10], rel) for i, rel in enumerate(ev)
        )
        agg[m] = a
        print(
            f"{m:<6} " + "  ".join(f"{a[f'r@{k}']:.3f}" for k in args.ks)
            + f"  {a['mrr@10']:.3f}"
        )

    # 补充：完全漏掉（R 里一块都没进 top-50）的问题数 + 单块 Dense 独有贡献
    print("\n=== 谁完全够不着证据（top-%d 内 R 命中为 0）===" % args.candidate_limit)
    for m in ranks:
        zero = sum(1 for i, rel in enumerate(ev) if recall_at(ranks[m][i], rel) == 0)
        print(f"  {m:<6} 全漏问题 {zero}/{len(pairs)}")

    print("\n=== 逐问题抽样（三家 top-10 明细 Chunk）===")
    for i, (q, rel) in enumerate(zip(queries, ev)):
        if i >= 6:
            break
        print(f"\n  Q: {q[:90]}")
        for m in ranks:
            hits = [c for c in ranks[m][i][:10] if c in rel]
            print(f"    {m:<5} top10 命中 {len(hits)}: {['…'+c.split('_')[-1] for c in hits]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())