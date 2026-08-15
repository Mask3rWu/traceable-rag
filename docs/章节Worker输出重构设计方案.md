# 章节 Worker 输出重构设计方案

> 状态：设计草案（未实现）
> 适用范围：src/research 下章节 Worker 输出契约（ChapterSubmission / ResearchPacket）
> 说明：本文档合并了原《章节Worker输出字段设计分析.md》的分析结论，原文档已删除。

## 1. 背景与动因（简要）

当前章节 Worker 输出是“三层结构”：ContentBlock 正文 + Claim 事实层 + Decision 规则层。评估批次（`b_20260811_1957_backfill`）中暴露了三个问题：

1. **重复表达**：正文本质是 Decision 的散文复述；Claim 层又独立写一遍事实句，形成三套文本。
2. **脱链与凑数**：Claim 大量未被 Decision 引用（q6 26/70、q7 15/80、q8 22/80），Decision 的 claim_ids 经常为空；模型在顶满字段上限。
3. **成本高**：37 次 submit_chapter 失败中，14 次缺 required decisions、14 次正文超预算、8 次跨层对齐失败。每一次失败都带更长上下文重跑，是 supervisor 题耗时 2000s+ 的重要来源。

worker 输出重构与“优化方向 A（减少 LLM 调用/上下文）”是同一根因的两端：方向 A 处理输入侧（调用次数、上下文体积），本方案处理输出侧（单次生成多少、要过多少校验）。两者必须一起做。

## 2. 业务推导与可信条件（简要）

“让人信服”需要被三类角色接受，对应不同的可信要求：

| 角色 | 需要什么 |
|---|---|
| 现场评估人员 / 指挥员 | 可操作、无歧义、覆盖所有组合的规则 |
| 评审专家 | 可追溯、来源与设计分离、假设/验证透明、无内部矛盾 |
| 后续工程化 / MLLM | 机器可读的规则与编码 |

由此沉淀出六条可信条件：**可追溯、来源与设计分离、不确定性透明、自洽、完整可操作、受控术语/版式统一**。当前三层结构在“可追溯”和“受控术语”上有部分支撑，但存在两个结构性问题：

1. **审计元数据没进交付物**：逐规则证据、置信度、假设、验证要求大多只存在于 Claim/Decision 元数据，最终标准正文看不到，读者感受不到价值。
2. **跨章一致性靠模型自觉**：Decision + glossary 是自由文本，机器只能校验 ID 存在，校验不了“第 2 章出现‘四级毁伤’算不算违约”，导致 q6/q7/q8/q10 一致性系统性判 C。

本方案的目标就是：**保留可信性支撑，去掉重复表达，把审计信息下沉为机器可校验的结构化字段。**

## 3. 设计目标

1. **去掉重复表达**：规则文本只在 prose 写一遍，不再有独立的 Claim 层和 Decision.statement。
2. **保留可信性**：逐规则证据、依据类型、跨章契约仍然存在，但下沉到结构化字段。
3. **减少模型负担与重试**：schema 更小、跨层对齐校验消失、required decisions 改为更简单的契约引用。
4. **让跨章一致性可校验**：引入契约机制，一致性审查从“靠模型自觉”变成“机器可检查”。

## 4. Worker 输出最小形态

```
ChapterWorker 输出
{
  "status":  "sufficient | insufficient",
  "prose":  "<单块交付正文>",
  "rules": [
    {
      "basis":            "source | designed | synthesized",
      "evidence_ids":     ["E1", "E2"],        // 只给 alias，系统回填全文
      "rationale":        "<一句话>",          // 可选，默认省略
      "contract_id":      "ch03_threshold"     // 可选：引用某个跨章契约
    }
  ],
  "contracts": [
    {
      "contract_id":      "ch01_damage_levels",
      "type":             "terms | threshold | classification",
      "canonical_terms":  ["轻度毁伤", "中度毁伤", "重度毁伤"],
      "applies_to_chapters": ["ch02", "ch03"]
    }
  ],
  "gaps":       ["待标定/待验证项..."],
  "conflicts":  [...]   // 可选：仅真正需要上抛的来源矛盾
}
```

### 4.1 字段说明

| 字段 | 是否必填 | 说明 |
|---|---|---|
| `status` | 是 | `sufficient` / `insufficient`。`insufficient` 必须由运行时证据覆盖等结构化条件支撑，不再依赖模型自由文本自评。 |
| `prose` | 是 | 单块交付正文，最终进入标准文档。不允许内含来源名、证据 ID、内部契约 ID。 |
| `rules[]` | 建议有 | 有序规则清单，表示本章需要“可追溯/可复用/可校验”的规则。**不是每条句子都写**，只写契约级/判据级规则。 |
| `rules[].basis` | 是 | 规则依据类型：`source`（直接转述来源）、`designed`（设计值/新规则）、`synthesized`（跨来源综合）。 |
| `rules[].evidence_ids` | 是 | 引用已注册证据（E1/E2 alias），系统回填全文。 |
| `rules[].rationale` | 否 | 该证据支持该规则的一句话理由。 |
| `rules[].contract_id` | 否 | 引用本交付物中的某个跨章契约。 |
| `contracts[]` | 否 | 本章“颁布”的跨章契约/术语轴。 |
| `contracts[].contract_id` | 是 | 全局唯一的契约 ID。 |
| `contracts[].type` | 是 | 契约类型：`terms`（受控术语）、`threshold`（阈值）、`classification`（等级/分类）。 |
| `contracts[].canonical_terms` | 依 type | `terms` 类型必填，只列规范词，不列禁用别名。 |
| `contracts[].applies_to_chapters` | 否 | 声明该契约适用的下游章节。 |
| `gaps` | 否 | 只放“待标定/待验证/外推确认”类内容披露，最终进入交付物 limitations。 |
| `conflicts` | 否 | 仅当来源确实矛盾且需要上抛给评审时填写，不再写“已自行消化”的矛盾。 |

> 注：`confidence / assumptions / validation_requirements` **首版不引入** rules[]，取舍见 §6。

### 4.2 规则锚点机制（逐规则注解的可行性方案）

不在 prose 内嵌 `[R1]` 等标记，避免污染公开正文、避免模型做脆弱的 span 对齐。

- 模型按规则在 prose 中出现的顺序输出 `rules[]`；
- 运行时只校验每条 `evidence_ids` 是否属于本章证据池、`contract_id` 是否合法；
- 组装器把 `R1..Rn` 渲染成独立的“规则依据表”（audit 视图），不进入公开正文；
- 评审专家通过该表逐条溯源，公开文档保持干净。

### 4.3 契约机制

`contracts[]` 与 `contract_id` 是同一机制的两面：

- **`contracts[]`**：产出章节“颁布”的跨章约定（定义）；
- **`contract_id`**：消费章节/规则“引用”该约定（使用）；
- 下游章节用 `required_contracts` 声明需要哪些契约。

示例：

```json
// 第 1 章颁布
"contracts": [
  { "contract_id": "ch01_damage_levels", "type": "classification",
    "canonical_terms": ["轻度毁伤", "中度毁伤", "重度毁伤"] }
]

// 第 3 章声明消费
"required_contracts": ["ch01_damage_levels"]

// 第 3 章某条规则引用
"rules": [
  { "evidence_ids": ["E10"], "basis": "synthesized",
    "contract_id": "ch01_damage_levels" }
]
```

一致性审查可据此做三件事：

1. 校验 `required_contracts` 的 ID 是否有上游章节产出；
2. 校验 `rules[].contract_id` 是否都在 `required_contracts` 内；
3. 扫描消费章节正文，出现的等级/术语词是否都在契约的 `canonical_terms` 内，不在即报违约。

### 4.4 gaps 语义收敛

- 保留：内容性披露，如“R 权重为规范设计值，需试验与仿真标定”。
- 移除：过程性自评（“check_terminology 返回 52 项疑似词……”）和“请重跑我”的证据缺口自报。
- 重跑/停止决策改由运行时结构化条件判断（证据覆盖、契约满足、结构校验），不再读模型自由文本。

### 4.5 与当前字段映射

| 当前字段 | 重构后去向 |
|---|---|
| ContentBlock | 保留为 `prose` |
| Claim | 删除，并入 `rules[].evidence_ids + rationale` |
| Decision.statement | 删除，规则文本只在 prose 写一遍 |
| Decision 元数据（confidence/assumptions/validation） | 首版不引入；由 `basis` + `gaps` 承担披露（见 §6） |
| Decision.glossary + produces/required | `contracts[] + contract_id + required_contracts` |
| Conflict | 保留但可选，且只上抛真正的矛盾 |
| gaps | 收敛为“待标定/待验证项” |
| diagnostics | 移出模型 schema，由运行时填失败原因 |

## 5. 校验规则（草案）

1. `prose` 非空，且不含来源名、证据 ID、内部契约 ID、Markdown 标题。
2. `rules[].evidence_ids` 必须全部属于本章（及上游共享）证据池。
3. `rules[].contract_id` 必须出现在 `required_contracts` 或本章 `contracts[]` 中。
4. `contracts[].contract_id` 在本交付物内唯一；`type=terms` 时 `canonical_terms` 非空。
5. `gaps` 只允许“待标定/待验证”类内容，不允许诊断性自评。
6. `conflicts` 仅当确有未消化矛盾时出现；已自行消化的矛盾不再上抛。
7. 不再有“ContentBlock 必须引用每个 Claim/Decision 恰好一次”“evidence 必须同时被 Claim 和 Decision 解释”等跨层对齐校验。

## 6. 关于 confidence / assumptions / validation_requirements 的取舍

这三个字段在旧设计里是 `DecisionRecord` 的元数据，作用是“不确定性透明”。是否下沉到 `rules[]`，权衡如下。

### 6.1 加入的理由（Pro）

1. **机器可读的审计**：结构化字段可被下游程序直接消费（如低置信度预警、自动生成验证清单），而 `gaps` 是散文，程序化消费弱。
2. **逐规则粒度**：能精确知道“哪一条规则是设计值、需要验证”，不依赖读者读整个章节的 prose。
3. **可强制透明**：可加校验“`basis=designed` 的规则必须带 validation_requirements”，防止设计值被写成来源事实。
4. **面向后续工程化**：README 提到下游 MLLM/工程化需要机器可读规则，结构化字段更利于复用。

### 6.2 不加入的理由（Con）

1. **与 gaps 重复**：当前 eval 里诚实披露已经在 gaps/prose/limitations 充分体现，且 judge 已认可（诚实性普遍 A）。再在每个 rule 里写一遍置信度/假设/验证，是重复输出。
2. **增加模型负担与输出体积**：每个字段都增加 completion token、schema 校验失败面和重试概率，与优化方向 A 直接冲突。
3. **消费方缺失**：目前这些字段没有被最终交付物消费，属于“纯支出无收益”。
4. **难可靠校验**：confidence/assumptions/validation 基本是自由文本/主观量，机器很难校验，容易退化成模型“填满字段”的凑数项——与当前 10 claim / 4 decision 顶满上限同类。
5. **新的不一致源**：不同章节对同一设计值的 confidence 可能不一致，反而给跨章一致性引入新的校验负担。

### 6.3 建议

**首版不加入这三个字段到 `rules[]`**，理由是不加入的收益（减负、去重、少校验）更契合当前最痛的成本问题，且没有真实消费方来体现其价值。

替代方案：

- **必填 `basis`**：已能区分 `source / designed / synthesized`，这是“来源与设计分离”的最小机器可读信号；
- **`gaps` 承担“待标定/待验证”披露**：作为进入交付物 limitations 的内容，保持诚实性；
- 若后续出现真实消费方（如规则引擎要自动生成验证清单、MLLM 要按置信度过滤），再按需引入一个轻量的 `design_notes` 或 `validation` 字段，且**只对 `basis=designed` 的规则开放**，避免全量扩散。

## 7. 与优化方向 A/B 的协同

- **输出侧（本方案）**：减少单次 completion、减少重试、缩小上下文。
- **输入侧（方向 A）**：证据 slot 化、每章检索预算、稳定 prompt 前缀。
- **检索侧（方向 B）**：planner 预生成查询集，按 `contract_id` 预取证据池。

三者不是二选一，建议一起推进。

## 8. 落地顺序与验证实验

1. 先按本 schema 重构 `ChapterSubmission`，删除 Claim/Decision 跨层对齐校验。
2. 再叠加方向 A 的上下文压缩与检索预算。
3. 用 q6/q7/q8 三个 supervisor 题重跑，对比以下指标：

| 指标 | 现值（参考） | 目标 |
|---|---:|---:|
| submit_chapter 失败 | 37 次（全批） | 显著下降，跨层对齐类归零 |
| supervisor 模型调用 | 88~116 次/题 | 40~60 次/题 |
| 末尾 prompt tokens | 5万~10万 | 显著下降 |
| elapsed_s | ~2000s/题 | 600~900s/题 |
| judge consistency/operability | C | 目标 ≥ B |

## 9. 风险与注意点

1. **不要砍掉审计信息**：`rules[].evidence_ids/basis` 与 `contracts[]` 必须保留；`confidence/assumptions/validation` 不引入后，**“待标定/待验证”披露必须通过 `gaps` 保留**，否则会丢失当前 eval 的诚实性优势。
2. **rules[] 不要写成“逐句注解”**：只写契约级/判据级规则，否则会重新变重。
3. **契约粒度先保守**：首版只做 `terms`（受控术语）和 `classification`（等级口径）两类，`threshold` 可后置。
4. **兼容回填**：历史 run 仍是旧 schema，读旧数据时保留兼容层，不阻塞新格式上线。
