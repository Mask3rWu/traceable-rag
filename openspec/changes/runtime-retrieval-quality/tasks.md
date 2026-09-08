## 1. 零成本检索质量模块

- [x] 1.1 新建 `src/research/retrieval_quality.py`，实现"从 run dict 抽取弱金"：收集 `answer.evidence_ids` 与 `worker_packets[].evidence_ids`，join 到 `evidence[]`（snapshot 均用 `ev-…` id），取每个被采纳证据**最后一次** `retrieval` trace 的 `dense_rank`/`bm25_rank`/`final_rank`。验证：对 `eval/runtime/batches/b_20260817_0028/runs/q7.json` 跑通，能列出被采纳证据及其单路 rank。
- [x] 1.2 实现三类指标函数（生产深度取自 `retrieval_top_k`，不硬编码）：路由贡献（`dense_rank < bm25_rank` 占比与反向）、单路救援率（`bm25_rank>深度且 dense_rank≤深度` → `rescue_by_dense`，反向 → `rescue_by_bm25`，两路都 >深度 → `fusion_only` 单列）、两路一致/冗余（两路均 ≤深度 占比）。验证：对 0028 全部成功 run 计算，能被之前手工脚本的数值（TOTAL cited=114、denseAhead≈52%、rescue_by_dense≈25%、rescue_by_bm25≈22%）复现到误差内。
- [x] 1.3 单测覆盖边界：`failed`/无 result run 剔除、evid 缺单路 rank 时跳过而非计 0、两路都 >深度 只归 `fusion_only`。验证：`pytest tests/test_retrieval_quality.py` 通过（12 passed）。

## 2. 接入批次流程

- [x] 2.1 在 `scripts/eval_runtime.py` 汇总阶段调用该模块，为每个含可用 run 的批次产出检索质量块（路由贡献/救援/一致 + 被采纳证据数）。验证：对 `b_20260817_0028` 重跑或对既有 `runs/` 汇总，得到与模块单测一致的批级数字（answered 114）。
- [x] 2.2 检索质量块**加性**写入 `q*.json` 与 `summary.md`（新增字段/新行），不改动既有字段与结局分类。验证：`openspec` 批次既有字段 diff 为空，质量块字段新增。
- [x] 2.3 对 0028（或最近一个含成功 run 的批）做端到端冒烟：summary.md 出现检索质量行，失败的 q6/q8/q9 不参与，数值与 2.1 一致。验证：质量表仅含 q1/q2/q5/q7/q10（completed），failed/routed 排除。

## 3. 离线扫权脚本

- [x] 3.1 新建 `scripts/analyze_retrieval_sweep.py`：参数为批目录（或 query/gold 文件）→ 抽取真实 query 与弱金（chunk 级，经 `ChunkCatalog`）；失败/无弱金 query 剔除；提供 `--max-queries` 上限。验证：对 0028 抽取 query 数≈103、弱金映射无缺失。
- [x] 3.2 单次检索：对抽取 query 跑 `DenseRetriever.search_many` + `BM25Retriever.search_many` 各一次（`candidate_limit` 与生产深度对齐），缓存候选；对权重网格（含基准 1:1）循环 `reciprocal_rank_fusion(..., limit=深度)`，判弱金 chunk_id 在 fused top-深度命中率。验证：重构实现=每 query 单次检索、权重网格复用候选；权重循环/深度命中以合成候选通过（命中受 top-K 门控、同集多权重复用）。注：#live sweep 需 pgvector+embedding 可达（本机 shell 现为连接超时）。
- [x] 3.3 输出权重→top-深度命中对照表（文本 + 可选 JSON），标注查询数、生产深度、弱金口径，明确"决策依据、不改配置"。验证：对照表含基准 1:1 行；live run 待检索栈可达（DB 当前未连通）。
- [x] 3.4 所有权重结果不写生产状态、不改 `RetrievalOptions`/`retrieval_top_k`。验证：git diff 仅新增报告/脚本文件，检索配置无改动；检索失败时脚本报清晰信息后正常退出、不留半成品。

## 4. 校验与收尾

- [x] 4.1 `openspec validate --change runtime-retrieval-quality` 通过（spec 场景格式、至少一 delta 能力）。
- [x] 4.2 依项目约定归档/收尾：相关只读分析说明写进 `docs/notes/`（代码 docstring 只引章节语义、不写文件名），提交走 conventional commits。验证：`git status` 无未说明文件，提交信息符合规范。（提交动作保留，等用户确认后执行。）