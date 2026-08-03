# 领域研究 Agent

当前实现是研究 Agent 的第一个可运行纵向切片。它复用现有 Dense、BM25 与 RRF，
不实现模型 Reranker，也不依赖前端 UI。关键阶段由代码中的显式工作流约束，模型只
负责检索词规划和基于给定证据的结构化综合。

## 运行流程

```text
问题
  -> 规划检索词
  -> 逐词混合检索
  -> 解析并去重 Evidence
  -> 生成 Claim / Conflict
  -> 核验来源、块归属与精确引文
  -> 原子写入 run.json
```

每次状态切换和每轮检索后都会更新 `run.json`。失败运行也会保留，`status` 为
`failed`，`error` 记录异常类型和信息。默认路径为：

```text
processed/research/runs/<run_id>/run.json
```

## DeepSeek 配置

Agent 和其他项目配置一样，只读取项目根目录 `.env`。DeepSeek 提供
OpenAI-compatible Chat Completions 接口，可配置为：

```dotenv
RESEARCH_MODEL=deepseek-chat
RESEARCH_BASE_URL=https://api.deepseek.com
RESEARCH_API_KEY=<your-key>
RESEARCH_MAX_QUERIES=4
RESEARCH_EVIDENCE_LIMIT=10
```

`RESEARCH_MODEL` 可以替换为服务端实际开放的其他 DeepSeek 模型名。模型密钥不会
写入 `run.json`、工具调用记录或日志。Embedding 和 PostgreSQL 仍读取已有的
`EMBEDDING_*` 与 `DB_*` 配置。

## 执行

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
- `ToolCall`：查询规划、每轮搜索、综合和引用核验的输入与结果摘要。
- `ResearchRun`：完整状态、上述对象、摘要、错误和时间信息。

引用核验目前验证内容哈希、文档、页码、块归属和引文是否存在于原文中。它不判断
“原文是否在语义上足以推出结论”，这需要后续独立的语义蕴含评测和人工审核。

来源展示 UI 后续直接读取 `run.json`，不参与来源事实的生成或核验。
