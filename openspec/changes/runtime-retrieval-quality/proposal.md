## Why

当前 `eval/runtime` 批次评估对检索只统计"量"——检索次数、去重、去重引用覆盖（`compute_retrieval`），没有任何"质"的刻画。因此我们**没有生产真实数据支撑两个关键决策**：Dense 与 BM25 谁在真正干活、混合检索（RRF）值不值、当前 1:1 等权是否合理。而 run 轨迹里其实已经埋了判断所需的全部原料（每条被检索证据的 `dense_rank` / `bm25_rank` / `fusion_score` + 被采纳证据即弱金），可以从已有数据零成本算出"哪一路在救哪类证据"，再做一次廉价的权重扫描来决定 RRF 权重。补上这一环，批次评估才能回答"这一批检索质量如何"，也才能基于真实分布而非模板评测题来定权重。

## What Changes

- 在 `eval/runtime` 批次流程（`scripts/eval_runtime.py`）中**新增"检索质量"维度**：对每个批次从 `runs/*.json` 抽取真实 query 与被采纳证据（弱金），计算并按题/按批汇总：
  - **路由贡献**：被采纳证据中，Dense 与 BM25 各自把证据排得更好（`denseAhead`/`bm25Ahead`）的比例；
  - **单路救援率**："单靠某一路上限内会漏、靠对方/融合救回"的被采纳证据占比（`rescue_by_dense` / `rescue_by_bm25`）；
  - **两路一致/冗余**：两路都在 top-N 内的比例（任一路单用即够）vs 两路都深、仅靠融合入池的比例；
  - 该题批次检索质量写在 `q*.json` 与 `summary.md` 中，持续随批次产出。
- **新增离线扫权脚本**：对真实 query 一次性检索（Dense 全候选 + BM25 全候选，不重复 embedding），扫描一组 RRF `dense_weight`/`bm25_weight`，测量"弱金进入生产深度（top-8）的命中率随权重的变化"，输出权重→质量对照表，用于**决策** RRF 权重。
  - 只产出决策依据，**不自动改生产权重**；是否落权重由后续显式变更决定。
- 检索质量全部为**只读分析**：不修改生产检索逻辑、不写入 chunk 数据、不影响研究 Agent 运行。失败/无证据的 run 被过滤，不参与统计。

## Capabilities

### New Capabilities
- `research/retrieval-quality`：从 run 轨迹与（可选的）离线扫权，量化检索的"路由贡献 / 单路救援 / 一致性 / 权重健康度"，为 Dense vs BM25 重要性判断与 RRF 权重调整提供生产真实数据依据。

### Modified Capabilities
<!-- 无既有能力需要等级别行为变更：检索质量是新增、只读、加性的分析维度，不改变 run 结局分类或现有批次产出的既有字段。 -->

## Impact

- 代码：`scripts/eval_runtime.py`（batch 汇总新增检索质量块）、新增一个检索质量分析模块（预置于 `src/` 或 `scripts/`，复用 `DenseRetriever` / `BM25Retriever` / `reciprocal_rank_fusion` / `evaluate_rankings`）、`summary.md` 与 `q*.json` 产物新增检索质量字段。
- 依赖：需要连 pgvector 与 embedding 端点（Dense 检索已有）；BM25 为本地 CPU；无新增第三方依赖。
- 系统：不改生产检索行为、不改 run 存储 schema（读取现有字段即可）；新增字段为加性、向后兼容。
- 成本：一次性离线——真实 query 每批约百条级（0028 实测 103 条），Dense embedding 一次 + BM25 一次；扫权重复用候选、纯 CPU，可忽略。