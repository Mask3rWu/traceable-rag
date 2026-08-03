# 领域研究 Agent

当前主入口是基于 LangGraph/LangChain 的路由 ReAct Agent。它复用现有 Dense、
BM25 与 RRF；模型自主决定检索词、证据读取、补充检索和停止时机。旧的固定纵向
工作流仍保留在 `scripts/run_research.py`，用于兼容和对照。

统一入口先由 Router 判断任务复杂度：

```text
请求 -> Router -> Fast ReAct (聚焦问答)
             \-> Supervisor ReAct (复杂研究/标准生成)
                    -> delegate_research -> Worker ReAct
```

只有 Supervisor 可以创建 Worker。Worker 只能检索、读取证据并提交结构化
`ResearchPacket`，不能继续创建子 Agent。最终运行结果默认写到：

```text
processed/research/agent-runs/<run_id>/run.json
```

## ReAct 运行约束

- `search_knowledge` 只向模型返回来源元数据和短预览；正文通过 `read_evidence` 按需读取。
- Fast Agent 至少成功检索一次，并声明有效 Evidence ID 后才能提交答案。
- Supervisor 只有收到带核验 Claim/Evidence 的 `sufficient` Worker 结果后才能提交标准。
- Worker 的消息上下文和正文读取预算相互独立，Evidence 注册表在一次请求内共享并去重。
- Langfuse callback 从根图透传到 Router、Supervisor、Worker、模型和工具调用。
- 不保存模型隐藏思维链；本地 `run.json` 保存最终答案、Evidence 和 Worker 研究包。

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
RETRIEVAL_DEFAULT_TOP_K=8
RESEARCH_MAX_EVIDENCE_READS=12
RESEARCH_MAX_WORKERS=4
RESEARCH_MAX_SUBTASKS=8
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

启用 Langfuse 后，一次用户请求对应一条根 trace。Router、Fast/Supervisor、
`delegate_research`、Worker、LLM 和检索工具均作为其下 observations 上报。

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
- `Claim`：结论文本、`direct/synthesized/hypothesis` 类型及精确引文。
- `Conflict`：相关结论与证据、处理状态和可选解决说明。
- `ResearchPacket`：Worker 的任务状态、Claim、Conflict、证据 ID 和证据缺口。
- `AgentRun`：路由决策、最终答案、去重 Evidence 和所有 Worker 研究包。

引用核验目前验证内容哈希、文档、页码、块归属和引文是否存在于原文中。它不判断
“原文是否在语义上足以推出结论”，这需要后续独立的语义蕴含评测和人工审核。

来源展示 UI 后续直接读取 `run.json`，不参与来源事实的生成或核验。
