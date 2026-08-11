# 章节 Worker 输出字段设计分析

> 样本：`processed/research/agent-runs/d8314fe3ffac4959a3a697a97bb359cf/run.json`
> 请求：生成一份坦克毁伤评估标准（normative_synthesis，7 章）
> 本报告只做设计层分析，结论为“基于单一 run 的强假设”，需更多样本验证。

## 1. 结论先行

当前系统的核心目标可以概括为：**输出一份有依据、让人信服的标准**。但“让人信服”目前没有可操作化的验收标准，这本身就是当前设计最先要解决的问题。

从业务需求反推，章节 Worker 输出的合理最小形态是**两层**：

- **交付正文（prose）**：标准真正面向读者的规则文本；
- **依据注解（annotations）**：每条规则 → 引用的 RAG 片段（evidence）+ 理由（rationale）+ 类型/置信度/假设/验证等元数据。

当前设计的核心问题不是“字段太多”，而是**同一知识被三层文本重复表达**（Claim 文本、Decision statement、ContentBlock 散文三套），并且这些审计元数据**没有进入最终交付正文**，导致“重而不管用”。当前设计也并非没有价值——它产出证据链、置信度、假设、验证要求、术语表、冲突、gap 等可信性支撑，问题在于表达与消费方式。

## 2. 业务 / 需求推导（候选，需确认）

“让人信服的标准”没有现成指标，以下推导属于**假设**，需要业务方确认。

### 2.1 谁需要被说服

| 角色 | 需要什么 |
|---|---|
| 现场评估人员 / 指挥员 | 可操作、无歧义、覆盖所有组合的规则 |
| 评审专家 | 可追溯、来源与设计分离、假设/验证透明、无内部矛盾 |
| 后续工程化 / MLLM | 机器可读的规则与编码 |

### 2.2 候选“可信条件”及当前 run 的表现

| 可信条件 | 含义 | 本 run 实际表现 |
|---|---|---|
| 可追溯 | 每条规则能定位到 RAG 片段 + 理由 | 部分满足：证据链完整（82 条），但粒度是“整章”，正文不含任何来源/证据标注，读者无法逐条溯源 |
| 来源与设计分离 | 能区分“文献怎么说”与“本标准怎么定” | 元数据上有 `direct/synthesized` vs `normative`，但 28 个 decision 全部为 `normative`，部分本质是转述来源；最终正文无区分 |
| 不确定性透明 | 设计值/推算值标注置信度、假设、验证要求 | 元数据满足（confidence/assumptions/validation），但正文基本不呈现（“置信度”仅出现 4 次、“设计值”仅 1 次、“假设/替代”0 次） |
| 自洽 | 章内与跨章无矛盾 | 不满足：最终仍有 5 个一致性 issue（3 error + 2 warning） |
| 完整可操作 | 覆盖目标场景，无未定义组合 | 不满足：issue_5 指出运动/火力判定表未覆盖非运动/火力分系统重度组合 |
| 受控术语 / 版式统一 | 术语、字段、编码一致 | 部分满足：有 glossary（6 轴），但 issue_3 暴露 P 类型同时覆盖防护与乘员的重叠 |

### 2.3 关键判断

当前输出的“元数据层”价值真实存在，但**几乎没有被最终交付物消费**：标准正文里看不到逐规则证据、置信度、假设与验证要求。也就是说，系统花大力气生成的可信性信息，最终读者看不到——这是“重”的直接原因之一。

## 3. run 现状：字段实际效果

### 3.1 数据总览

| 指标 | 值 |
|---|---|
| 章节 / packet | 7 / 7（全部 sufficient） |
| Evidence 注册表 | 567 条 |
| 实际引用（packet = block = answer 一致） | 82 条 |
| Claim | 70（direct 49 / synthesized 21） |
| Decision | 28（全部 normative） |
| Conflict | 2（均 resolved） |
| 一致性 issue | 5（3 error + 2 warning） |

### 3.2 正文与 Decision / Claim 的相似度

按章节取 ContentBlock 与各 Decision 语句、各 Claim 语句的最大字符相似度：

| 章节 | 正文长度 | 与 Decision 最大相似度 | 与 Claim 最大相似度 |
|---|---|---|---|
| ch1 | 622 | 0.34 | 0.09 |
| ch2 | 735 | 0.33 | 0.11 |
| ch3 | 576 | 0.27 | 0.07 |
| ch4 | 676 | 0.42 | 0.09 |
| ch5 | 812 | 0.35 | 0.12 |
| ch6 | 616 | 0.49 | 0.09 |
| ch7 | 582 | 0.24 | 0.05 |

含义：**正文本质就是把 Decision 写成散文**（相似度 0.24~0.49），与 Claim 几乎无重叠（0.05~0.12）。模型实际上把规则写了两遍（Decision.statement + 正文），Claim 又作为第三层事实陈述存在。

### 3.3 字段脱链 / 反例

- **ch6**：4 个 decision 全部 `claim_ids=[]`，10 个 claim 全部未被任何 decision 引用——在最需要“出处”的质量控制章节，Claim 层反而完全脱链。
- **ch4 / ch7**：各有 1 个 claim（`CH4-C3`、`C5`）未被 decision 引用。
- Decision 引用 Claim 的个数分布 `{0:4, 1:1, 2:6, 3:9, 4:6, 5:1, 6:1}`，说明模型对“decision 由 claim 推导”的建模很不稳定。
- 证据引用虽是闭合的（packet = block = answer = 82），但粒度是整章，不是逐规则。

### 3.4 gaps / diagnostics / conflicts 的实际内容

- **gaps** 绝大多数是“设计值需标定”，而非“缺资料”：如 20%/50% 时速阈值、10/30 分钟时限、80% 一致率、A 系数、代码表码元分配等。
- **diagnostics** 混入了多种性质的内容：执行失败原因、`check_terminology` 误报说明、字符预算说明、一致性修订说明——它不是单一语义。
- **conflict** 仅 2 条且全部被模型自行消化为 resolved，说明模型倾向内部消化来源矛盾，而不是把矛盾上抛给评审。

## 4. 合理最小设计的推导

### 4.1 从可信条件反推需要哪些字段

| 可信条件 | 需要的字段 |
|---|---|
| 可追溯 | 每条规则的锚点 + `evidence_ids` + `rationale` |
| 来源与设计分离 | 每条规则的 `basis`（source_derived / designed / synthesized） |
| 不确定性透明 | designed 规则带 `confidence / assumptions / validation_requirements` |
| 自洽 | 结构校验可基于“规则清单”，不需要 Claim/Decision 两层对齐 |
| 完整可操作 | 正文规则完整 + 覆盖矩阵（评审/校验用） |
| 受控术语 | 术语契约：`glossary` + `applies_to_chapters` |

### 4.2 推导出的最小形态（推荐方向）

```
ChapterWorker 输出
├─ prose              # 单块正文，最终交付
├─ annotations[]      # 逐规则依据注解
│   ├─ anchor         # 指向 prose 中某条规则（正文唯一锚点）
│   ├─ basis          # source_derived | designed | synthesized
│   ├─ evidence_ids[] # 引用的 RAG 片段
│   ├─ rationale      # 为何该证据支持该规则
│   ├─ (仅 designed) confidence / assumptions / validation_requirements
│   └─ (可选) contract_id / applies_to_chapters / glossary
├─ conflicts[]        # 可选：仅当来源确实矛盾且未消化时
├─ gaps[]             # 改为“待标定 / 待验证项”
└─ diagnostics[]      # 仅运行时失败原因，不由模型填充自检说明
```

### 4.3 与当前设计的映射

| 当前字段 | 推导后去向 |
|---|---|
| ContentBlock | 保留为 `prose` |
| Claim | **并入 annotations**（事实层变成规则的 `evidence + rationale`，不再单独产出一层文本） |
| Decision.statement | **删除**（规则文本只在 prose，避免写两遍） |
| Decision 元数据（confidence / assumptions / validation / glossary / applies_to_chapters） | 保留，下沉到 annotation 的契约字段 |
| Decision 跨章节契约（produces / required） | 用 annotation 的 `contract_id` 表达 |
| Conflict | 保留但可选，且明确“上抛给评审”的职责 |
| gaps | 语义收敛为“待标定 / 待验证项” |
| diagnostics | 只保留运行失败原因 |

## 5. 当前设计 vs 推导设计：优缺点

| 维度 | 当前设计 | 推导设计 |
|---|---|---|
| 追溯粒度 | 章节级证据集合 + decision 级 rationale | 逐规则锚点 |
| 表达成本 | Claims / Decisions / 正文三层文本 | 单一正文 + 注解 |
| 模型负担 | 10 claim + 4 decision 预算强迫多写 | 只写正文 + 依据注解 |
| 审计价值 | 高，但未进交付物 | 保留并下沉到规则 |
| 一致性审查 | 输入丰富但需两层对齐 | 直接基于规则清单 |
| 跨章节契约 | Decision + produces/required | annotation contract_id |
| 主要风险 | 不一致、脱链、成本高 | 需补“锚点”与“契约”机制设计 |

**当前设计优点**：可审计、可追溯、有跨章节契约、有受控术语与冲突/gap 记录，这些是“让人信服”的必要支撑。

**当前设计缺点**：同一知识被写三遍；Claim 层在标准生成链路里是多余的中间跳；审计元数据没有进入最终正文，读者感受不到价值；成本高且质量不稳定（脱链、5 个 issue）。

**推导设计优点**：保留上述可信性支撑，去掉重复表达，把“证据 + 理由”下沉到每条规则，模型负担更小、更好校验。

**推导设计代价**：需要设计“锚点”（正文规则 ↔ 注解的稳定映射）与“契约”（跨章节引用）机制；如果实现不当，可能丢失当前“独立事实层可复用于下游”的能力。

## 6. 结论与设计层建议

1. **保留并强化**可追溯、来源/设计分离、不确定性透明、受控术语、自洽校验——这些是“让人信服”的核心支撑。
2. **减重**：删除独立 Claim 层；Decision.statement 与正文去重；diagnostics 只作运行失败；gaps 收敛为“待标定 / 待验证项”。
3. **先明确业务**：最优先的不是继续堆字段，而是把“让人信服”转化为可验收条件（本报告 §2），并决定这些条件是否要呈现在最终标准正文里（例如逐规则置信度/来源标注）。
4. **开放问题**：正文规则与注解的锚点机制、跨章节契约的粒度、最终文档是否内嵌来源/置信度标记——需要业务与工程共同决策。

## 7. 数据来源与限定

- 本报告所有数字来自单一 run `run.json`，结论是“基于该样本的强假设”，需更多 run 验证。
- 相似度使用去空白字符的 SequenceMatcher 计算，仅作相对量级参考。
- 本报告不修改任何代码。
