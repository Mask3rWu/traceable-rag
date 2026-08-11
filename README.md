# 可溯源 RAG 与领域研究 Agent

## 1. 项目定位

本仓库是「视觉毁伤评估」完整项目的前置基础工作，负责在权威文献之上构建一套**可溯源的 RAG 与领域研究 Agent**，为后续的数据增强、模型微调和端到端实时评估系统提供可信的知识底座与准则来源。当前首个落地场景为**装甲目标视觉毁伤评估准则构建**。

本仓库覆盖两块业务内容：

- **可溯源 RAG**：面向约 47 份权威文献（国军标 + 领域论文，约 120 万字符），完成 PDF 解析、噪声过滤与结构化切块，提供混合检索与证据溯源核验。
- **准则生成**：基于路由式多智能体，按章节依赖调度 ReAct Worker 多轮取证，经一致性审查后确定性成文，准则可回溯到原文页码。

数据增强（图生图）、QLoRA 微调与端到端实时评估系统不在本仓库范围，它们在本仓库产出的可溯源知识库与评估准则之上独立开展。

## 2. 整体架构

系统按层组织，各层边界明确，不把「知识库构建」「检索」「准则生成」混为一个步骤：

```text
行业标准 / 技术规范 / 领域文献
        ↓
  原始资料与知识库   (解析、结构恢复、切块)
        ↓
  通用可溯源检索     (混合召回、来源数据契约)
        ↓
  领域研究 Agent     (ReAct, 证据与冲突持久化)
        ↓
  任务封装           (问答 / 深度研究 / 标准生成)
        ↓
  评估准则、机器规则、溯源报告
```

- **资料与知识库层**：PDF 解析、版面与章节恢复、块间关系、结构化切块。
- **检索层**：返回与问题相关、可回溯的原始证据，不直接生成准则。
- **研究 Agent 层**：ReAct 决定检索词与工具调用，关键过程由显式工作流约束；证据、结论、冲突持久化。
- **任务封装层**：在 Agent 之上封装问答、深度研究、标准生成三种模式。

## 3. 可溯源 RAG

### 3.1 数据预处理

解析层把原始 PDF 转为可溯源的结构化结果，只负责「读准、定位准、关系全」：

- **解析引擎**：PP-StructureV3，版面检测（PP-DocLayoutV3）、OCR（PP-OCRv5）、表格识别（SLANeXt，转 HTML）、公式识别（PP-FormulaNet，转 LaTeX）；图片/示意图（figure）不做图转表，直接裁剪交 MLLM 做语义理解。
- **块归一**：原始标签统一为 `heading / paragraph / table / formula / figure / caption / list / appendix`；为每个块分配 `block_id`、`section_path`（标题编号层级）、页码与归一化坐标。
- **关系补全**：`caption_of` / `caption_ids` 配对图表与图注；`references` 建立正文到图表/公式/章节的交叉引用；`continuation_of` / `continues_to` 连接跨栏跨页正文。这些关系在解析阶段一次建好，跨页引用也能命中。
- **噪声过滤**：剥离页眉、页脚、页码；GJB 橙色水印检测与弱化；低质量内容标记为 `unreadable` 而非补写。

切块（`chunks.jsonl`）以**章节为硬边界、语义单元为切分粒度**：标题-正文、图表-图注、跨页续接、公式-编号保持强绑定；`embedding_text`（含文档名与完整标题路径，用于向量化）与 `text`（纯正文）分离；每个 chunk 保留 `block_ids`、页码、`section_path`、`heading_path` 与图表回链，满足溯源到原文页码的需要。

### 3.2 混合检索

针对领域专有名词密集导致的向量检索漂移，采用 Dense + BM25 混合检索：

- **Dense**：`BAAI/bge-m3`（1024 维），pgvector cosine HNSW。
- **BM25**：Jieba 搜索模式分词，索引文档名 + 标题路径 + 正文（不索引 MLLM 生成的 `visual_text`）。
- **融合**：RRF（Reciprocal Rank Fusion），支持按通道加权；当前为等权基线。
- **一致性校验**：Dense、BM25 与本地 chunk 的 `content_hash` 必须一致，不一致直接失败，避免索引版本错配。
- Reranker 仅预留接口（`NoopReranker`），待独立评测集就绪后再接入。

### 3.3 评测与溯源核验

检索评估候选集（`eval/`）直接标注到解析层 `doc.json` 的稳定 `block_id`，不依赖 `chunks.jsonl`，可公平比较不同切块策略。指标含 `evidence_recall`、`hit_rate`、`complete_recall`、`MRR`。

在 47 篇文档、3779 个 chunk、2083 道问题上（Dense/BM25 各召回 50 条，等权 RRF，`rank_constant=60`，`eval/rag/v2`）：

| 方法 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| Dense | 0.8195 | 0.8661 | 0.7066 |
| BM25 | 0.9009 | 0.9353 | 0.8064 |
| 等权 RRF | 0.8781 | 0.9288 | 0.7844 |

> 早期 `eval/rag/v1` 的批量题字面泄漏明显、偏向词法检索，BM25 单路排序偏高；等权 RRF 的优势主要体现在更深的候选集（Recall@20/50 最高），适合为后续 Reranker 提供覆盖。当前为中性基线，融合权重待独立评测集就绪后再定。

证据溯源由数据契约约束：`Evidence` 持稳定 ID、内容哈希、文件、页码、章节、块 ID 与原文；引用核验验证 Evidence ID 存在且通过 `content_hash / 页码 / 块归属 / 原文片段包含` 核验，模型提交的引文不作为可信输入。

## 4. 路由式多智能体准则生成

### 4.1 路由与编排

统一入口先由 Router 判断任务复杂度，分流到两条路径：

```text
请求 -> Router -> Fast ReAct (聚焦问答)
             \-> Chapter Planner (章节与依赖 DAG)
                    -> Chapter Scheduler (按依赖分波次并发)
                         -> Chapter Worker ReAct (章节级研究)
                    -> Consistency Reviewer (结构一致性审查)
                    -> Deterministic Assembler (确定性组装成文)
```

- **章节依赖 DAG**：`DocumentPlan` 的 `depends_on` 描述章节间依赖，带环检测与决策可达性校验；基础章节负责术语、范围、分级体系等全局决策。
- **分波次并发**：同一依赖层的章节并发执行，上游证据不足时下游不盲目启动；Worker 不能继续创建子 Agent。
- **首轮失败回退**：章节首轮证据不足或结构提交失败时，调度器把明确 `gaps` 反馈给同一 Worker 做一次补充研究，仍未完成才阻塞下游。

### 4.2 证据与引用机制

研究产出按三级解耦，保证准则可回溯：

- **Claim**：证据支持的事实，类型为 `direct / synthesized / normative / hypothesis`。
- **DecisionRecord**：基于一个或多个 Claim 形成的规则，含依据、假设、替代方案与验证要求。
- **ContentBlock**：章节正文。一个完成章节必须且只能提交一个正文块，章节标题与编号由组装器统一生成。

引用链由 `Claim.citations` 表达：**Worker 只提交 Evidence ID，系统从已核验证据中回填精确引文**，不要求模型逐字复述原文。核验验证 Evidence ID 存在且通过哈希、页码、块归属与原文包含检查。

### 4.3 术语一致性

为缓解多章节下术语漂移（如毁伤程度两套体系并存），引入受控术语表：

- 基础章节产出结构化 `glossary`（仅规范词，不含禁用别名），下游章节在 `required_glossary` 中声明消费，作为前馈约束注入。
- Worker 在 `submit_chapter` 前可调用只读的 `check_terminology` 自检，按术语轴动态生成匹配模式列出疑似漂移词；该工具是**建议性**的，永不阻塞提交、不触发重跑。
- 匹配模式从 glossary 动态生成，不硬编码任何领域词汇，换研究任务自动适配。

### 4.4 一致性审查与确定性成文

- **一致性审查**：检查正文块关联、Claim/Decision 覆盖与正文引用等结构问题，作为可见性上报。
- **确定性组装**：纯代码无 LLM，按章节计划拼接正文块、章节标题与编号，禁止照搬来源文档目录编号，并受全文与单章字数、Claim、Decision 上限约束。

### 4.5 产物与观测

一次请求的完整结果写入 `processed/research/agent-runs/<run_id>/run.json`，含路由、章节计划、一致性问题、答案、去重 Evidence 与章节研究包。标准生成另产出三件套：`standard.md`（人工阅读）、`rules.json`（机器可读规则）、`traceability.json`（规则到文档/章节/页码/块的映射）。

全链路观测接入 Langfuse：一次用户请求对应一条根 trace，Router、Planner、Worker、Reviewer、Assembler、LLM 与检索工具均作为其下节点级 observation 上报，用于节点级排错与异常定位。
