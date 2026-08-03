# 多 Agent 研究任务故障观察记录（2026-08-03）

## 观察范围

- 测试输入：`生成一份坦克毁伤评估标准`
- 当前跟踪任务：`350aa6faf2eb4425afb284c737c4d308`
- 同输入对照任务：`3a77384078814510ba5a32fab57e6d6a`
- 先前同类任务：`a86f73bae9bd4934864dd10bf44b2a9f`
- 模型：`deepseek-v4-flash`
- API：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:5173`

本文只记录可从 API 事件、持久化产物和当前代码确认的现象。时间均为 Asia/Hong_Kong。

## 当前结论

任务不能稳定完成并非单一检索故障，而是至少包含三条彼此独立的失败路径：

1. 取消只改变展示状态，不会停止 graph；任务显示“停止中”后仍会继续调用工具和启动后续章节。
2. Planner 的长 JSON 输出偶发不符合 `DocumentPlan`，且规划阶段没有重试，单次解析错误会直接终止整次运行。
3. Chapter Worker 偶发不调用 `submit_chapter`，或提交的 artifact 缺少计划要求的字段；连续两轮无有效提交后，章节失败并阻塞整个依赖链。

## 问题 1：显示“停止中”但研究继续执行

### 现象

任务 `350aa6faf2eb4425afb284c737c4d308` 在 21:28:39 收到取消请求，事件明确记录：

```text
cancel_requested
interruptible: false
Cancellation requested; the active provider call cannot be interrupted
```

此后任务没有停在当前 provider 调用结束处，而是继续发生：

- 21:29:31：进入 `chapter_research`。
- 21:29:31：启动第一章 worker。
- 21:29:36 以后：持续调用 `search_knowledge`、`read_evidence`。
- 21:34:32：调用 `submit_chapter`，因缺少决策失败。
- 21:35:48：再次提交第一章并成功。
- 21:35:48：继续并发启动两个下游章节。

因此前端显示“停止中”与后端实际运行状态不一致。这里不仅是“无法中断正在进行的 provider 请求”，而是取消后完全没有阻止后续 graph 节点、worker 和工具调用。

### 代码原因

- `RunManager.cancel()` 在活动任务无法由 `Future.cancel()` 取消时，只把状态设置为 `cancel_requested` 并发送事件。
- 取消状态或取消令牌没有传给 `RoutedResearchAgent.run()`、`AgentRuntime.run()` 或工具层。
- graph 的节点边界和工具调用前没有取消检查。
- `RunManager._execute()` 在 graph 最终返回后，会无条件用 `result.outcome` 覆盖当前状态。因此任务还可能从“停止中”直接跳为“已完成”或“未完整生成”，丢失取消语义。

涉及代码：

- `src/api/manager.py`：`cancel()`、`_execute()`。
- `src/research/service.py`：`RoutedResearchAgent.run()` 未接收取消信号。
- `src/research/graph.py`：根 graph、worker graph 和工具调用路径均未检查取消状态。

### 期望行为

最低可接受语义应是：当前不可中断的 provider 请求返回后，在下一个 graph 节点或工具调用前检测取消信号，终止调度，并将最终状态固定为 `cancelled`。不能继续启动新章节，也不能用研究结果覆盖取消状态。

## 问题 2：Planner 结构化输出解析失败后整次任务终止

### 复现

对照任务 `3a77384078814510ba5a32fab57e6d6a` 的时间线：

```text
21:28:39 running
21:28:41 route_selected: supervisor
21:28:41 chapter_planner
21:29:21 failed: OutputParserException
```

模型输出中至少有两处确定的 schema 错误：

- 第 3 章把必填字段 `acceptance_criteria` 写成 `acceptance_cence_criteria`。
- 第 7 章的 `research_questions`/JSON 尾部残缺，导致该章同样缺少 `acceptance_criteria`。

Pydantic 因此报告两个 `DocumentPlan` 校验错误，任务在启动任何 chapter worker 前失败。

### 代码原因

- Planner 使用 `with_structured_output(DocumentPlan, method="json_mode")`。`json_mode` 只约束返回 JSON，不等于服务端严格执行 Pydantic schema。
- `plan_document()` 只调用一次 `planner.invoke()`，没有针对 `OutputParserException` 的纠错重试、缩短计划或降级策略。
- `ChatOpenAI` 初始化未显式配置输出 token 上限和请求超时。长计划的尾部残缺风险未被隔离。

### 附带的可观测性问题

- 事件回调的 `on_chain_error()` 只清理内部状态，不发出 `stage_failed` 事件。
- 失败任务的 API 摘要里 `route` 为 `null`，尽管此前已经产生 `route_selected: supervisor`。原因是摘要只从最终 `AgentRun` 读取 route，而 Planner 异常时没有最终结果。
- 顶层异常只保存在进程内 `ManagedRun.error`；失败发生在 `AgentRun` 持久化之前，API 重启后该失败详情可能丢失。

## 问题 3：Chapter Worker 无有效 artifact 或 artifact 校验失败

### 先前任务的致命失败

任务 `a86f73bae9bd4934864dd10bf44b2a9f` 成功完成 Router 和 Planner，但第一章两轮 worker 都只进行了检索与证据读取，最后没有产生可识别的 `submit_chapter` 调用。运行结果为：

```text
The chapter worker failed to submit a valid artifact.
Worker stopped without a valid submit_chapter call
```

第一章状态变为 `failed` 后，其余 7 章因上游章节或决策缺失全部变为 `blocked`，最终任务为 `incomplete`。

### 当前任务中的相似问题

当前任务第一章第一次确实调用了 `submit_chapter`，但 artifact 缺少计划要求的 `scope` 和 `terminology` 两个 `DecisionRecord`，工具校验失败。错误被回传给 worker 后，第二次提交补齐字段并成功，因此第一章没有失败。

“毁伤等级分类框架”章节的首轮 worker 则再次出现“研究完成后没有有效 `submit_chapter`”的模式，调度器于 21:41:09 启动第二轮补充研究。第二轮继续检索、读取证据，但最终仍未调用 `submit_chapter`，诊断为 `Worker stopped without a valid submit_chapter call`。该章节因此失败，并阻塞了依赖它的后续章节。

“评估数据与采集标准”章节已在 21:39:25 成功提交。

### 代码原因和放大因素

- worker 模型未返回 tool call 时，`next_step()` 无论是否还有 step 预算都会立即进入 `exhausted`；不会追加一条“必须调用提交工具”的纠错消息。
- 调度器只允许首轮加一轮补充研究。第二轮失败后直接形成失败 packet。
- `WORKER_PROMPT` 和章节输入仍要求为每条正文证据创建 `EvidenceContribution`，但当前 `ResearchPacket` schema 已不存在这个字段。提示词与工具 schema 冲突，会增加模型不提交或生成错误参数的概率。
- `submit_chapter` schema 较大，包含正文块、Claim、Decision、Conflict、引用和规范性决策约束；当前模型在长 tool-call 参数上的稳定性不足。
- 事件只显示工具校验错误；当模型返回普通文本而非 tool call 时，原始可见输出没有进入诊断，最终只能看到笼统的 `Worker stopped without a valid submit_chapter call`。

## 本次运行时间线（持续更新）

| 时间 | 事件 | 结果 |
| --- | --- | --- |
| 21:28:22 | 创建任务 | `running` |
| 21:28:23 | Router | `supervisor` |
| 21:28:39 | 请求取消 | 状态变为 `cancel_requested`，实际未停止 |
| 21:29:31 | Planner 完成并启动章节研究 | 继续执行，证明取消未生效 |
| 21:34:32 | 第一章首次提交 | 失败：缺少 `scope`、`terminology` |
| 21:35:48 | 第一章第二次提交 | 成功 |
| 21:35:48 | 启动两个并发章节 | `damage_classification`、`data_requirements` |
| 21:39:25 | 数据要求章节提交 | 成功 |
| 21:41:09 | 毁伤分类章节启动第二轮 | 首轮无有效提交，第二轮进行中 |
| 21:44:44 | 进入一致性检查和组装 | 毁伤分类章节第二轮仍无有效提交 |
| 21:44:54 | 任务结束 | `incomplete`；取消状态被结果状态覆盖 |

## 本次运行最终结果

- 最终状态：`incomplete`。
- 路由：`supervisor`。
- 证据数：377。
- worker packet 数：8。
- 成功章节：2 个，分别为 `scope_and_terminology`、`data_requirements`。
- 失败章节：1 个，为 `damage_classification`；诊断为 `Worker stopped without a valid submit_chapter call`。
- 阻塞章节：5 个，分别为 `assessment_methods`、`assessment_process`、`tools_validation`、`reporting`、`quality_improvement`。
- 关键缺失决策：`damage_levels`，并进一步导致 `assessment_methods`、`process_model`、`tool_validation_standard`、`report_format`、`improvement_protocol` 无法形成。
- 持久化产物：`processed/research/agent-runs/350aa6faf2eb4425afb284c737c4d308/run.json`。
- Trace ID：`b01647fb1ba351224cbdb115fe8ae636`。

本次结果同时确认了两个问题：第一，worker 无有效 artifact 的故障可稳定跨任务、跨章节复现；第二，任务在 `cancel_requested` 后仍执行到组装阶段，并最终被 `_execute()` 改写为 `incomplete`，前端“停止中”不是可靠的终止状态。

## 建议修复顺序

1. 修复取消语义：传递取消令牌，在 provider 返回后和每个节点/工具前检查；终态不得覆盖 `cancelled`。
2. 给 Planner 增加 schema 校验失败后的有限重试，并缩短/分段生成计划；记录原始完成原因和 token 使用信息。
3. 删除提示词中已经不存在的 `EvidenceContribution` 要求，确保 prompt 与 `ResearchPacket` schema 一致。
4. worker 返回非 tool call 且预算未耗尽时，追加明确纠错消息并再次调用模型，而不是立即 `exhausted`。
5. 持久化失败运行、阶段错误和 route；在 UI 中区分“已请求取消，等待安全停止”和“仍在执行新工作”的异常状态。
6. 增加真实模型回归用例：Planner 长 JSON、缺少决策后的修复提交、worker 普通文本返回、运行中取消及取消后的终态保护。
