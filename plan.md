# 术语一致性治理：worker 自检 + 结构化术语表决策

## 背景与定位

run 40be17e9（supervisor 路由，7 章全部 sufficient 但 outcome=incomplete）暴露的问题：
- 毁伤程度两套体系并存（轻度/中度/重度/歼灭 vs 轻微/中等/严重）。
- P/C 级适用范围在章节间不一致。
- 数据采集章节未记录"毁伤程度"字段。
- "杀伤等级/毁伤等级/毁伤程度"术语混用。

经证据确认根因：下游 worker **已经通过 `_upstream_context` 收到 `terminology` decision**，但其 statement 只是一句宣告（"术语间不得混用"），不是可执行清单，漂移照样发生。

经多轮讨论确定的方向（用户拍板）：

> worker 是 ReAct 模式，在最后一次 `submit_chapter` 之前，让它**自己先做一次术语核对**。基础章节沿用现有机制产出**结构化术语表决策**（只写规范词，不写禁用别名）。不设校验失败、不重跑——worker 自检发现漂移就在同一个 run 内改完再提交；自检没发现也不强制纠正。

## 为什么这个方向成立

1. **不重跑**：核对在 worker 自己的 ReAct 循环里完成，发现漂移当场改，不需要外层 `_run_chapter_with_followup` 重试、不需要 raise。
2. **不依赖领域硬编码**：术语表由基础章节运行时生成，换研究任务自动适配，不写死禁用词。
3. **避开"事后整篇重生成不可校验"的硬伤**：核对只改本章节正文，不是对 7 章做不可控重写。
4. **和现有架构同构**：术语表就是一个 `DecisionRecord.glossary`，走现有 `produces/required_decisions` 通道，不新增图节点、不新增传递路径。

## 改动设计

### A. 数据结构（`src/research/agent_models.py`）

新增 `GlossaryEntry`，并给 `DecisionRecord` 加可选 `glossary` 字段：

```python
class GlossaryEntry(BaseModel):
    """受控术语表中的一个轴（如"杀伤等级""毁伤程度"）。"""
    axis: str = Field(min_length=1)                     # 术语轴名
    canonical_terms: list[str] = Field(min_length=1)    # 规范词
    scope: str = ""                                     # 适用说明，可选

class DecisionRecord(BaseModel):
    ...  # 现有字段不变
    glossary: list[GlossaryEntry] = Field(default_factory=list)
```

`glossary` 默认空，向后兼容现有 decision 和全部测试。

`ChapterPlan` 新增可选字段：

```python
class ChapterPlan(BaseModel):
    ...  # 现有字段不变
    required_glossary: list[str] = Field(default_factory=list)  # decision_id 列表
```

语义与 `required_decisions` 平行：下游章节声明 `required_glossary: ["terminology"]` 表示"我需要术语表来约束用词"。

### B. planner 提示词（`src/research/graph.py` `PLANNER_PROMPT`）

在现有"Put shared terminology ... in foundational chapters"之后追加：
- 产术语的基础章节，其术语 decision 必须填 `glossary`，列出每个术语轴的规范词；
- 消费术语的章节，在 `required_glossary` 里声明对应 decision_id；
- `glossary` 只列规范词，**不列禁用别名**。

### C. worker 自检工具 `check_terminology`（`src/research/graph.py` `_build_chapter_graph`）

给 chapter worker 的工具列表新增一个**只读核对工具**。worker 写完正文草稿后调用它，工具返回"疑似漂移词"清单，worker 自行决定是否修改。

工具签名（确定性，不让模型做判断）：

```python
def check_terminology(content_blocks: list[dict], decisions: list[dict]) -> str:
    """对照术语表，列出正文里出现、但不在任何术语轴规范词集合内的疑似术语级词汇。"""
```

实现逻辑：
- 从传入的 `decisions` 里提取所有 `glossary`，合并出每个轴的 `canonical_terms` 集合。
- 对每个 `content_block.markdown`，按"轴关键词"扫描——匹配模式从 glossary 的 `axis` 名和 `canonical_terms` **动态生成**（不硬编码领域词）。例如 axis="杀伤等级"、canonical_terms=`["K级","M级",...]`，则生成"X级"模式，提取正文里所有"X级"词，检查是否在规范集内。
- 不在规范集内的，作为 `suspect_terms` 返回给 worker。
- 工具**永远成功返回**（返回 JSON，可能是空列表），不 raise、不阻塞提交。

关键：匹配模式从 glossary 动态生成，不出现"轻度/严重/K/M/F"等字面量在代码里。"严重/毁灭"这类词是否被扫描到，取决于它们是否匹配某个轴的模式——若不匹配则不被标记，符合"不写禁用词"的边界。

**worker prompt 增补**（`_build_chapter_graph` 的 prompt 拼接，接在现有 Hard output contract 之后）：

> Before calling submit_chapter, call check_terminology with your draft content_blocks and the upstream glossary decisions. If it returns suspect_terms, revise your prose to use only the canonical terms from the glossary axes, then call submit_chapter. check_terminology is advisory: it never blocks submission. If you judge a flagged term is not a controlled-vocabulary drift, you may keep it.

"advisory、never blocks" 是对"不重跑"底线的明文保障。

### D. `_run_chapter` 注入术语表（`src/research/graph.py:600` 的 request）

在现有 `request["upstream"]` 之外，新增 `request["glossary"]`，把上游 `required_glossary` 对应的 decision 的 `glossary` 平铺出来，让 worker 在输入里直接看到规范词表：

```python
glossary_decisions = [
    d.model_dump(mode="json")
    for pkt in self._ancestor_packets(chapter, completed)
    for d in pkt.decisions
    if d.decision_id in chapter.required_glossary and d.glossary
]
request["glossary"] = glossary_decisions  # 可能是空列表
```

这样 worker 即使不调 `check_terminology`，也能在输入里看到规范词表，起到"前馈注入"的作用。`check_terminology` 是在它之上的额外自检。

### E. `check_terminology` 如何拿到 glossary

worker 调用 `check_terminology(content_blocks, decisions)` 时，`decisions` 参数由 worker 从它收到的 `request["glossary"]` / `request["upstream"]["decisions"]` 里传进来。工具本身**不访问 workspace 状态**（保持 ReAct 工具无状态、可测试），所有数据从参数进。

## 不做的事（明确边界）

- **不新增图节点**：不插入 `terminology_normalize` 节点，不事后整篇重生成。
- **不加校验失败 raise**：`_validate_packet` 不因术语漂移而 raise、不触发 `_run_chapter_with_followup` 重试。漂移不改变 packet 的 status。
- **不写禁用词清单**：glossary 只有 `canonical_terms`，没有 forbidden_aliases。
- **不硬编码领域词**：不出现"轻度/严重/K/M/F"等字面量在代码里；匹配模式从 glossary 动态生成。
- **不动 reviewer**：现有 `_structural_consistency_issues` 和 `REVIEW_PROMPT` 保持不变，仍会报漂移为 issue（作为可见性），但不驱动重跑。
- **不动 fast 路径**：`_build_fast_graph` 不受影响，fast 模式无术语表需求。

## 受影响文件

| 文件 | 改动 |
|---|---|
| `src/research/agent_models.py` | 新增 `GlossaryEntry`；`DecisionRecord.glossary`；`ChapterPlan.required_glossary` |
| `src/research/graph.py` | `PLANNER_PROMPT` 增补术语表要求；chapter prompt 增补自检指令；`_build_chapter_graph` 增加 `check_terminology` 工具；`_run_chapter` 注入 `request["glossary"]` |
| `tests/test_research_graph.py` | 新增 `check_terminology` 工具行为测试；现有 `_ScriptedModel.bind_tools` 需让 worker 分支也带上新工具（脚本模型不实际调用，保持兼容） |

## 测试策略

1. **`check_terminology` 单元测试**（纯函数，不依赖模型）：
   - 空 glossary -> 返回空 suspect_terms，不报错。
   - glossary 有 `杀伤等级` 轴、canonical=`[K级,M级,F级,C级,P级]`；正文含"Q级" -> 被标记为疑似词；正文含"严重"且不匹配"X级"模式 -> 不被标记。
   - glossary 有 `毁伤程度` 轴、canonical=`[轻度,中度,重度,歼灭]`；正文含"轻微"——若匹配程度轴模式则被标记，否则不标记（确认"不写禁用词"的边界）。
2. **集成测试**（沿用 `_ScriptedModel` 模式）：worker 脚本里包含一次 `check_terminology` 工具调用 + 一次 `submit_chapter`，验证工具返回不阻塞提交、最终 packet 正常生成。
3. **回归**：现有测试全绿（`glossary`/`required_glossary` 默认空，向后兼容）。

## 残留风险与取舍（如实告知）

- **工具的有效性依赖 worker 主动调用**：prompt 要求它调，但 ReAct 模型可能跳过。若跳过，等价于退回"纯前馈注入"，漂移仍可能发生——但比现状多了 `request["glossary"]` 前馈，仍是净改善。
- **"疑似词"的识别依赖模式匹配**：从 glossary 动态生成的模式可能漏报（某些漂移词不匹配模式）。这是"不写禁用词"的必然代价，符合用户选择。
- **不检测 = 不保证消除漂移**：本方案是"降低漂移概率 + 让规范词更显眼地进入 worker 视野"，不是"漂移归零"。归零需要禁用词清单或事后重生成，两者用户均已否决。

## 验证方式

改完后用 run 40be17e9 同样的请求重跑一次 supervisor 路由，对比 `damage_criteria` 章节正文是否仍出现两套体系、`data_collection` 是否记录毁伤程度字段。由于不重跑机制不变，重点看漂移是否减少、issue 数是否下降。
