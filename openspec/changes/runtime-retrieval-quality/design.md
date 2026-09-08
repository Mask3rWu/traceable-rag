## Context

现有 `eval/runtime` 批次流程由 `scripts/eval_runtime.py` 驱动：每题经 API 跑完一个研究 Agent run，`run_one` 产生 `q*.json`（含 `compute_retrieval` 的次数/去重/引用等**量**指标），并把完整 `result` 备份到 `runs/q*.json`。

`runs/*.json` 里已经埋好全部检索质量的原料（见 proposal.md — Why）：
- 每个 evidence 的 `retrieval[]` trace 含 `dense_rank` / `bm25_rank` / `dense_score` / `bm25_score` / `fusion_score` / `final_rank`（来自 `src/research/evidence.py` 的 `RetrievalTrace`，落盘可见）；
- 被采纳(弱金)证据 id 直接存在于 `answer.evidence_ids` 与 `worker_packets[].evidence_ids`（均为 `ev-…`），无需经 alias 反查；
- evidence 含 `chunk_id`，可用 `ChunkCatalog.block_ids(chunk_id)` 映射到 block（`src/retrieval/catalog.py`）。

生产检索深度：`RetrievalService.search()` 最终由 `NoopReranker` 截断到调用方 `limit`，研究 Agent 侧默认 `retrieval_top_k=8`、上限 20（`src/config.py:156`）。因此**真正消费给下游的工作深度是 8**，「top-8 命中」才是对口的口径——这也是本能力用生产深度而非模板评测的 Recall@5/MRR@10 的原因（后者属于 `eval/rag` 模板检索评测约定，二者是不同度量，见下文 Goals/Non-Goals）。

## Goals / Non-Goals

**Goals:**
- 从批次 run 轨迹**零成本**算出"路由贡献 / 单路救援 / 两路一致"三类检索质量指标，按题并按时序汇总进 `q*.json` 与 `summary.md`。
- 提供**只读离线扫权**脚本：对真实 query 单次检索（Dense 全候选 + BM25 全候选），扫描 RRF 权重组合，输出"弱金 top-8 命中率随权重"对照表，作为 RRF 权重决策依据。
- 全部只读：不改生产检索逻辑、不改 run 存储 schema、不写 chunk 数据。

**Non-Goals:**
- 不自动改生产权重（是否落权重由后续显式变更决定；本变更只产出依据）。
- 不改动 `eval/rag` 模板检索评测及其 Recall@5/MRR@10 约定——那是独立度量。
- 不引入 LLM-judge 重标金证据；先以"被采纳证据"为弱金，绝对精度留待人工锚。
- 不保证覆盖知识库主题之外的真实分布（runtime query 均属军事/装备领域）。

## Decisions

### D1. 两个计算入口分开：批次内零成本指标 vs 批次外离线扫权
- **A. 批次内检索质量指标**（每次批次自动产出，零检索成本）：读取 `runs/*.json`，对每个 evidence trace 只做统计，不调用检索。挂在 `eval_runtime.py` 汇总阶段，产出质量块写入 `summary.md` 与新增机器可读字段。
  理由：原料已在 run 里，零成本、随批次自动累积，天然贴合"每个批次回答检索质量如何"。
- **B. 离线扫权**（按需、读-only CLI，独立一个脚本）：从选定批次的 run 抽取真实 query 与弱金，单次检索后扫权重。不并入批次主流程，避免每次批次都背上 Dense embedding。
  理由：扫权是"要不要调权重"的决策工具，频率低、有 embedding 成本，适合手动触发而非随批次。

### D2. 弱金与口径定义
- 弱金 = 实际被写入 `answer.evidence_ids` / `worker_packets[].evidence_ids` 的证据（其 `chunk_id`→`ChunkCatalog.block_ids` 成 block 集）。
- "深度内"判定使用生产深度 `retrieval_top_k`（默认 8，取自配置，不硬编码）。
- **代表性 trace**：每个被采纳 evidence 可能被多个 query 召回（多条 trace），路由贡献/救援指标采其**最后一次出现的 trace**（对应 agent 最终证据状态）的 `dense_rank`/`bm25_rank`。备选（按 query 逐条判）列为 Open Question，不改变可观测契约。

### D3. 指标三个族（B 由 A 借 if 需全量）——严格对表 spec 三条 Requirement
1. **路由贡献**：被采纳证据中 `dense_rank < bm25_rank`（Dense 排得更好）的比例 与反向。
2. **单路救援率**：`bm25_rank > 深度 且 dense_rank ≤ 深度` → `rescue_by_dense`；反向 → `rescue_by_bm25`；两路都 > 深度 → `fusion_only`（单独报，不归各单路）。
3. **两路一致/冗余**：两路都 ≤ 深度 的比例（任一路单用即够）与累计。

### D4. 扫界实现复用现有构件
- 对真实 query 集：`DenseRetriever.search_many(limit=candidate_limit)` + `BM25Retriever.search_many(...)` 各一次（**同一批候选），不重复 embedding**。
- 权重扫描：循环调用 `reciprocal_rank_fusion(..., dense_weight, bm25_weight, limit=深度)`，对 fused top-深度判弱金 chunk_id 是否命中。
- 亦可复用 `evaluate_rankings`（`evaluation.py:69`）做 `k=深度` 的 weak-gold recall——把 query+block 集塞进 `EvaluationCase`。二选一实现均可，推荐轻量命中正（chunk 级 hit）起步。
- 权重网格：`dense_weight`/`bm25_weight` ∈ {0.5, 1.0, 1.5, 2.0} 的互异组合（避免退化 0 权重），精度由任务细化；报告会同筛当前 1:1 作为基准行。

### D5. 采样与规模防护
- 真实 query 每批约百条（0028 实测 103 条去重），不必采样；脚本加 `--max-queries` 上限防止误触发大集。
- 失败/无证据 run（如 0028 的 q6/q8/q9 无 result、q3 无引用）整题剔除，不参与统计（对应 spec 的排除场景）。

### D6. 代码落点
- 纯统计函数：`src/research/retrieval_quality.py`（读 run dict → 指标），可单测。
- 批次接入：`scripts/eval_runtime.py` 汇总处调用该模块，把质量块并进 `_gather_summary` / `_write_markdown`（加性字段）。
- 扫权：`scripts/analyze_retrieval_sweep.py`（读批目录 → 抽取 query/弱金 → 检索 → 扫权 → 输出表/JSON）。用现有 `src/retrieval/*` 与 `evaluation.py` 的复用件，不新增第三方依赖。
- 枚举取值/深度来自 `src/config.py`（`retrieval_top_k` / `RetrievalOptions`），不散落硬编码。

## Risks / Trade-offs

- **[弱金自洽偏置]** 金证据来自"当时检索策略"能命中的证据，测绝对召回会整体偏高 → 缓解：所有结论以"相对对比"表述（哪路贡献高、哪个权重更好），不断言绝对召回率；后续可选抽 20–40 条人工 gold 当锚（Non-Goals 之外的未来项）。
- **[池外盲区]** run 只记录"被 fusion top-8 捞进池"的证据，**未进池的漏检无记录**，故救援指标只能描述"入池的被采纳证据"，量不到"该被采纳却从未入池" → 缓解：指标措辞限定"入池且被采纳"；扫权因检索全候选、可比批次内指标多覆盖一部分漏检盲区（仍受限于弱金）。
- **[代表性 trace 简化]** 多 query 召回同一证据时只取末次 trace，可能忽略早期高相关轮次 → 缓解：作为近似接受；如需精确可在此能力后续迭代为 per-(query,evidence)（Open Question），不构成本变更契约缺口。
- **[embedding/DB 依赖]** 扫权需要 pgvector 与 embedding 端点可用；不可用则扫权失败 → 缓解：脚本清晰报缺、失败不落半成品；批次内零成本指标(A)不依赖检索，始终可跑。
- **[触发成本]** 若把扫权误挂进批次主流程会每次背 Dense embedding → Non-Goal 明示，扫权独立脚本、手动触发。

## Migration Plan

- 纯加性、只读、无 schema 变更：批次产物 JSON 新增字段、`summary.md` 增一行质量块，既有字段/结局分类不动 → 无需数据迁移。
- 生产方式不变：批次照旧由 `eval_runtime.py` 产出，新增质量块仅在读取到可用证据时出现。
- 回滚：移除质量块调用即可，无副作用残留（不写生产状态、不改检索配置）。
- 落地权重（如扫权后决定改 `RetrievalOptions` 或 `retrieval_top_k`）属**后续独立变更**，不在本变更自动执行。

## Open Questions

- 路由贡献是否要升级为"per-(query,evidence)"逐条统计（而非取末次 trace）？——不改变 spec 的可观测结论方向，可延后决定。
- 扫权网格的精确取值与是否报告"逐 query 明细"？——实现细节，不影响本变更任务划分。
- 扫权将来是否并入批次流程（改为每次批次自动附带）？——频率与成本权衡，留待后续评审。