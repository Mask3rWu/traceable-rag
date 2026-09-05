"""诊断：每个 eval query 上，dense 与 bm25 对证据块的各自排名。

只读分析工具，不改任何生产逻辑。用法：
    python scripts/diag_gold_ranks.py [--eval-root eval/rag/v2] [--limit N] [--show SAMPLE]
输出四分类汇总 + 抽样明细，帮定位 Dense 输在召回还是排序。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROJECT_ROOT as ROOT  # noqa: E402
from src.retrieval.bm25 import BM25Retriever, DEFAULT_INDEX_ROOT  # noqa: E402
from src.retrieval.catalog import ChunkCatalog  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402


@dataclass
class Gold:
    case_id: str
    question: str
    category: str
    block_ids: list[str]     # gold 证据 block
    chunk_ids: list[str]    # 解析成 chunk；无则不可达


def load_cases(root: Path) -> list[Gold]:
    golds: list[Gold] = []
    for path in sorted(root.glob("*.jsonl"), key=lambda x: x.name):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            golds.append(
                Gold(
                    case_id=f"{path.stem}:{row['question_id']}",
                    question=row["question"],
                    category=row.get("category", "?"),
                    block_ids=list(row["evidence_block_ids"]),
                    chunk_ids=[],
                )
            )
    return golds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=ROOT / "eval" / "rag" / "v2")
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--limit", type=int, help="截断 query 数（调试）")
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--output", type=Path, help="把所有行写 TSV（可读明细）")
    args = parser.parse_args(argv)

    catalog = ChunkCatalog.load()
    # block_id -> 包含它的 chunk_id（召回判定与 evaluation 对齐：chunk.block_ids）
    block_to_chunks: dict[str, set[str]] = defaultdict(set)
    for cb in catalog._sources:
        for bid in catalog.block_ids(cb):
            block_to_chunks[bid].add(cb)

    golds = load_cases(args.eval_root)
    for g in golds:
        chunks: set[str] = set()
        for bid in g.block_ids:
            chunks |= block_to_chunks.get(bid, set())
        g.chunk_ids = sorted(chunks)
    if args.limit:
        golds = golds[: args.limit]
    queries = [g.question for g in golds]
    if not queries:
        raise SystemExit("no questions")

    print(f"检索 {len(golds)} queries ...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    dense_results = DenseRetriever().search_many(queries, limit=args.candidate_limit)
    bm25_results = BM25Retriever(DEFAULT_INDEX_ROOT).search_many(
        queries, limit=args.candidate_limit
    )
    print(f"检索耗时 {time.perf_counter()-t0:.1f}s", file=sys.stderr, flush=True)

    dense_best: list[int | None] = []
    bm25_best: list[int | None] = []
    for g, dr, br in zip(golds, dense_results, bm25_results):
        if not g.chunk_ids:  # 不可达：gold chunk 不在当前库
            dense_best.append(None)
            bm25_best.append(None)
            continue
        drmap = {r.chunk_id: r.dense_rank for r in dr if r.dense_rank is not None}
        brmap = {r.chunk_id: r.bm25_rank for r in br if r.bm25_rank is not None}
        dense_best.append(min((drmap[c] for c in g.chunk_ids if c in drmap), default=None))
        bm25_best.append(min((brmap[c] for c in g.chunk_ids if c in brmap), default=None))

    # 分类：both / bm25_only / dense_only / both_miss / unreachable
    by_bucket = Counter()
    by_cat: dict[str, Counter] = defaultdict(Counter)
    rows: list[tuple] = []
    for g, dr, br in zip(golds, dense_best, bm25_best):
        if not g.chunk_ids:
            key = "unreachable"
        elif (dr is None) and (br is None):
            key = "both_miss"
        elif br is not None:
            key = "both" if dr is not None else "bm25_only"
        else:  # dr is not None, br is None
            key = "dense_only"
        by_bucket[key] += 1
        by_cat[g.category][key] += 1
        rows.append((g, dr, br, key))

    print(f"\n=== 四分类汇总（recall 上限 = top-{args.candidate_limit}）===")
    for k in ["both", "bm25_only", "dense_only", "both_miss", "unreachable"]:
        print(f"  {k:<12} {by_bucket[k]:>5}")

    print("\n=== 按题型 × 分类 ===")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(
            f"  {cat:<26} both={c['both']:<5} bm25_only={c['bm25_only']:<5} "
            f"dense_only={c['dense_only']:<5} miss={c['both_miss']:<5}"
        )

    both = [(g, dr, br) for g, dr, br, k in rows if k == "both"]
    if both:
        drs = [dr for _, dr, _ in both]
        brs = [br for _, _, br in both]
        dense_worse = sum(1 for _, dr, br in both if dr > br)
        print(f"\n=== both 集 n={len(both)}：dense 是否系统性排更靠后 ===")
        print(f"  dense best 中位 {statistics.median(drs):.0f} 均值 {statistics.mean(drs):.1f}")
        print(f"  bm25  best 中位 {statistics.median(brs):.0f} 均值 {statistics.mean(brs):.1f}")
        print(f"  dense rank>bm25（dense 更靠后）: {dense_worse} ({dense_worse/len(both)*100:.1f}%)")
        for k in (10, 20, 50):
            print(f"  dense 在 top-{k} 内: {sum(1 for d in drs if d <= k)}/{len(both)}")

    print("\n=== 抽样（优先展示 dense 更靠后 / bm25_only 的典型失败）===")
    ordering = {"bm25_only": 0, "both": 1}
    shown = 0
    for g, dr, br, key in sorted(
        rows, key=lambda r: ordering.get(r[3], 9)
    ):
        if key == "unreachable":
            tag = "[unreachable]"
        elif key == "both_miss":
            tag = "[both_miss]"
        elif key == "bm25_only":
            tag = f"dense=miss bm25#{br} [bm25_only]"
        elif key == "dense_only":
            tag = f"dense#{dr} bm25=miss [dense_only]"
        else:
            tail = " [dense更靠后]" if dr > br else ""
            tag = f"dense#{dr} bm25#{br}{tail}"
        print(f"  {g.category:<20} | {tag:<38} | {g.question[:56]}")
        shown += 1
        if shown >= args.show:
            break

    if args.output:
        with args.output.open("w", encoding="utf-8") as fh:
            fh.write("category\tquestion_id\tdense_best\tbm25_best\tbucket\tquestion\n")
            for g, dr, br, key in rows:
                fh.write(
                    f"{g.category}\t{g.case_id}\t"
                    f"{'' if dr is None else dr}\t{'' if br is None else br}\t"
                    f"{key}\t{g.question}\n"
                )
        print(f"明细写入 {args.output.resolve()}")

    # 仅打印抽样（balance 便于阅读，不重复汇总）
    return 0


if __name__ == "__main__":
    raise SystemExit(main())