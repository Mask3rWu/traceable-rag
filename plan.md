# 移除 EvidenceContribution 第二层 + 放宽读取预算

## 目标
1. 证据链只保留 `Claim.citations` (claim -> evidence_id + quote);移除 `EvidenceContribution` 这层冗余的"采用理由"中间结构及其全部交叉闭合校验。
2. 放宽读取预算,先把项目跑通:`max_evidence_reads` 20→40、`worker_max_steps` 18→30。

## 原则
- 旧 run.json 是测试数据,不作兼容;`EvidenceContribution` 字段从 schema 和 UI 彻底删除。
- 不动检索层(dense/bm25/fusion)、不动 graph 拓扑(路由/规划/执行波次/审查/组装的节点结构不变)、不动 `Claim`/`Decision`/`ContentBlock` 其余字段。
- `Citation.quote` 仍由模型生成 + `verify_claim` 逐字校验——这一项本轮**不动**(属于上一轮讨论的"数据库回填"话题,独立改动,不在本次范围)。

---

## 改动点

### A. `src/research/agent_models.py` — 删除模型
- 删除 `EvidenceContribution` 类(整个 class,行 121-136)。
- `ResearchPacket` 删除字段 `evidence_contributions: list[EvidenceContribution]`(行 175)。
- `__init__.py` 的 `__all__` 若列了 `EvidenceContribution` 则移除。

### B. `src/research/graph.py` — 简化 `_validate_packet`(核心)
当前 `_validate_packet`(335-463)中 EvidenceContribution 相关校验全部删除:
- 删除 `contribution_pairs` 构建与全部基于它的校验(375-400、419-429)。
- 删除 `used` 的 `evidence_contributions` 来源(357-360 中的 contribution 部分)——`used` 改为只从 `claims.citations`、`decisions.evidence_ids`、`content_blocks.evidence_ids` 汇总。
- 删除 "missing_contributions" / "unexposed_contributions" / "Evidence contribution references unknown claim/decision/content block" / "Only context evidence may be linked solely to a content block" 这几类 ValueError。
- 删除 decision.evidence_ids 必须有 contribution 记录的校验(431-440 中 contribution 部分)——保留 decision.claim_ids 合法性校验。

保留的校验(不变):
- chapter_id / chapter_title 匹配(336-339)。
- claim/decision/block ID 唯一(344-349)。
- cited evidence 全部存在 + `validate_evidence_ids(used)` + `verify_claim`(351-373)。
- content_block 引用合法 claim/decision、且必须暴露 claim 或 decision(402-408)。
- normative decision 的 assumptions/alternatives/validation_requirements(431-449)。
- sufficient 章节的 prose+claim+证据+产出决策要求(451-462)、非 sufficient 不得有 prose(461-462)。

### C. `src/research/__init__.py` — 导出
- `__all__` 移除 `EvidenceContribution`。

### D. `src/config.py` — 放宽预算
- `ResearchModelConfig.max_evidence_reads` 默认 20 → 40(行 143)。
- `ResearchModelConfig.worker_max_steps` 默认 18 → 30(行 140)。
- `from_env` 中 `RESEARCH_MAX_EVIDENCE_READS` default=40(行 195)、`RESEARCH_WORKER_MAX_STEPS` default=30(行 188-189)。
- 注:`fast_max_steps` 默认 `min(max_steps, 8)`、`supervisor_max_steps` 默认 `max_steps`,基于 max_steps=12 自动派生,无需单独改。
- `.env.example` 同步更新示例值(60、57 行附近的注释块)。

### E. `tests/test_research_graph.py` — 更新测试
- 删除两处 `evidence_contributions` 测试数据(355-366、498-509)。
- `test_chapter_rejects_evidence_without_body_contribution`(399-442):该校验逻辑已被移除,删除该测试用例。
- 其余断言不变(`[ev-test]` 仍由 content_block.evidence_ids 注入,answer.evidence_ids 不受影响)。

### F. 前端 `web/src/types.ts` + `web/src/App.tsx`
- `types.ts`:删除 `EvidenceContribution` 接口(100-109)、`ResearchPacket.evidence_contributions` 字段(143)。
- `App.tsx`:
  - 删除 `contributionLabels`(22-25)。
  - 删除 import 中的 `EvidenceContribution`(10)。
  - `ChapterResearch` 组件:删除 `contributions` 取值与 `contribution-list` 渲染块(94、110-123),改为空状态提示或直接移除该区块;删除未再使用的 `Link2` 图标 import(若仅此处用)。
  - 行 248 的 tab 标签 `selectedPacket?.evidence_contributions?.length` 计数:改为统计 `claims` 或移除计数。

### G. `docs/research-agent.md` — 文档同步
- 行 35-36、126 关于 EvidenceContribution / "证据贡献理由" 的描述删除或改写为"证据链由 Claim.citations 表达"。

---

## 不改动(显式边界)
- 检索层(`retrieval/*`)、`EvidenceWorkspace.search/read` 行为。
- `read_evidence` 预算耗尽时的报错行为(本轮只放宽数字,不改软信号——那是独立改动)。
- `Citation.quote` 模型生成 + 逐字校验(独立改动,本轮不动)。
- graph 节点拓扑、DAG 调度、consistency review、assembler。

## 验证
- `python -m pytest tests/test_research_graph.py -q` 通过。
- 前端 `web` 构建无类型错误(`tsc`/vite build)。
- 可选:用真实请求跑一次 supervisor 确认不再出现 `Evidence contribution ...` 类报错。
