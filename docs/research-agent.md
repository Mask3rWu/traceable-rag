# 领域研究 Agent

当前主入口是基于 LangGraph/LangChain 的路由 ReAct Agent。它复用现有 Dense、
BM25 与 RRF；模型自主决定检索词、证据读取、补充检索和停止时机。旧的固定纵向
工作流仍保留在 `scripts/run_research.py`，用于兼容和对照。

统一入口先由 Router 判断任务复杂度：

```text
请求 -> Router -> Fast ReAct (聚焦问答)
             \-> Chapter Planner (章节与依赖 DAG)
                    -> Chapter Scheduler (按依赖分波次并发)
                         -> Chapter Worker ReAct (章节级研究)
                    -> Consistency Reviewer
                    -> Deterministic Assembler
```

复杂任务先生成结构化 `DocumentPlan`。基础章节负责术语、范围、分级体系等全局
决策；依赖章节只有在上游完成并产生所需 `DecisionRecord` 后才会执行。同一依赖层
的章节可以并发，Worker 不能继续创建子 Agent。请求生成新标准时，计划使用
`normative_synthesis` 模式：先完成资料的总结、比较、冲突核验和适用性分析，再基于
这份证据综合设计新的标准。它不是跳过 `evidence_summary` 的捷径，而是“证据综合 +
规范性设计”的组合交付。不要求资料中已经存在一份完全相同的成品标准。新规则必须
标记为 `normative`，并公开参考依据、迁移理由、假设、替代方案和验证要求。
`evidence_summary` 则表示证据综合本身就是最终交付物。最终运行结果默认写到：

```text
processed/research/agent-runs/<run_id>/run.json
```

## ReAct 运行约束

- `search_knowledge` 只向模型返回来源元数据和短预览；正文通过 `read_evidence` 按需读取。
- Fast Agent 至少成功检索一次，并声明有效 Evidence ID 后才能提交答案。
- 每个章节 Worker 先提交对相关来源的总结、比较和核验结果；在 `normative_synthesis` 模式下，再提交与来源事实分离的规范性设计。Worker 提交正文块、核验 Claim、Decision 和公开推断。
- 证据链由 `Claim.citations` 表达：Worker 只提交 Evidence ID，系统从已核验证据中回填精确引文，不再要求模型逐字复述原文。
- 全局决策通过依赖关系传递给后续章节；上游证据不足时，下游依赖章节不会盲目执行。
- 首轮章节研究证据不足或结构提交失败时，调度器会把明确的 `gaps`/`diagnostics` 反馈给同一章节 Worker 做一次补充研究；仍未完成才阻塞下游。阻塞章节不会再次启动 Worker，也不会继续调用一致性模型。
- 每个 Worker 会收到完整章节目录作为内容所有权边界，但只能提交当前章节。一个完成章节必须且只能提交一个 `ContentBlock`，块标题为空且正文不能包含 Markdown 章节标题；章节标题与编号由组装器统一生成，禁止照搬来源文档的目录编号。
- 整份文档默认最多 6000 个正文字符，并按计划章节数平均分配；每章还受 1600 字符硬上限，以及 10 个 Claim、4 个 Decision 的上限约束。超出预算的 `submit_chapter` 会被拒绝并要求压缩重写；这些限制可通过 `RESEARCH_DOCUMENT_MAX_CHARS` 和 `RESEARCH_CHAPTER_MAX_*` 配置调整。
- Claim 与 Decision 是多对多关系：Claim 记录证据支持的事实，Decision 记录基于一个或多个 Claim 形成的规则。正文块中的每个 Evidence ID 必须由 Claim citation 或 Decision rationale 给出使用原因，且正文块必须关联本章全部 Claim、Decision 和 Evidence。
- Worker 的消息上下文和正文读取预算相互独立，Evidence 注册表在一次请求内共享并去重。
- Langfuse callback 从根图透传到 Router、Planner、章节 Worker、Reviewer、模型和工具调用。
- 不保存模型隐藏思维链；只保存可审计的依据、公开推断、假设和替代方案。

## DeepSeek 配置

Agent 和其他项目配置一样，只读取项目根目录 `.env`。DeepSeek 提供
OpenAI-compatible Chat Completions 接口，可配置为：

```dotenv
RESEARCH_MODEL=deepseek-chat
RESEARCH_BASE_URL=https://api.deepseek.com
RESEARCH_API_KEY=<your-key>
RESEARCH_MAX_QUERIES=4
RESEARCH_EVIDENCE_LIMIT=10
RESEARCH_MAX_STEPS=12
RESEARCH_FAST_MAX_STEPS=8
RESEARCH_WORKER_MAX_STEPS=18
RESEARCH_SUPERVISOR_MAX_STEPS=12
RETRIEVAL_DEFAULT_TOP_K=8
RESEARCH_MAX_EVIDENCE_READS=20
RESEARCH_MAX_WORKERS=4
RESEARCH_MAX_SUBTASKS=8
RESEARCH_DOCUMENT_MAX_CHARS=6000
RESEARCH_CHAPTER_MAX_CHARS=1600
RESEARCH_CHAPTER_MAX_CLAIMS=10
RESEARCH_CHAPTER_MAX_DECISIONS=4
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

`RESEARCH_MAX_STEPS` 是兼容回退值。Fast 和 Chapter Worker 默认分别使用 8、18 个
模型决策步骤；Worker 需要多轮搜索和正文读取，因此预算高于快速问答。
`RESEARCH_SUPERVISOR_MAX_STEPS` 当前作为协调器兼容配置保留；章节规划、一致性审查和
确定性组装各执行一次，不进入旧的 Supervisor ReAct 循环。

启用 Langfuse 后，一次用户请求对应一条根 trace。Router、Planner、Fast/Chapter
Worker、Consistency Reviewer、Assembler、LLM 和检索工具均作为其下 observations 上报。

`RESEARCH_MODEL` 可以替换为服务端实际开放的其他 DeepSeek 模型名。模型密钥不会
写入 `run.json`、工具调用记录或日志。Embedding 和 PostgreSQL 仍读取已有的
`EMBEDDING_*` 与 `DB_*` 配置。

## 执行

安装 Agent 依赖：

```powershell
conda run -n dba-py311 python -m pip install -r requirements-agent.txt
```

统一入口：

```powershell
conda run -n dba-py311 python scripts/run_agent.py `
  "生成一份可溯源的装甲目标视觉毁伤评估标准"
```

## Legacy 固定工作流

以下命令仅用于兼容和回归对照。其产物仍写到
`processed/research/runs/<run_id>/run.json`：

确保 Dense 和 BM25 索引已经构建，然后运行：

```powershell
conda run -n dba-py311 python scripts/run_research.py `
  "装甲目标的视觉毁伤等级应如何划分？"
```

可以临时覆盖查询数和每轮证据数：

```powershell
conda run -n dba-py311 python scripts/run_research.py `
  "装甲目标的视觉毁伤等级应如何划分？" `
  --max-queries 3 --evidence-limit 8
```

命令行会打印每条已核验结论及其 evidence ID，再打印 evidence ID 对应的文件、页码和
章节。完整引文、检索分数与工具轨迹保存在 `run.json`。

## 数据契约

- `Evidence`：稳定 evidence/chunk ID、内容哈希、文件、页码、章节、块 ID、原文、
  视觉资产和每轮检索排名。
- `Claim`：结论文本、`direct/synthesized/normative/hypothesis` 类型、Evidence ID 及系统回填的精确引文。
- `Conflict`：相关结论与证据、处理状态和可选解决说明。
- `DocumentPlan`：章节目标、研究问题、依赖关系、产出/使用的全局决策和验收条件。
- `ResearchPacket`：章节正文块、Claim、Decision、Conflict 和证据缺口。
- `AgentRun`：路由、`completed/incomplete` 结果状态、章节计划、一致性问题、答案、去重 Evidence 和章节研究包。

引用核验目前验证 Evidence ID 存在且证据已通过内容哈希、文档、页码、块归属和原文核验；模型提交的引文不会作为可信输入。它不判断
“原文是否在语义上足以推出结论”，这需要后续独立的语义蕴含评测和人工审核。

来源展示 UI 后续直接读取 `run.json`，不参与来源事实的生成或核验。
