# Proposal: 结果分类与观测落地

## Why

后续重试/降级、证据剪枝、续跑、索引调整都需要可比基线，但当前运行时观测基本不可留用：
失败在 `api/manager.py` 被裸 `except` 统一落成 `failed`；LangChain 事件轨（`events.py` → `manager.py` 的 `managed.events`）**纯内存、SSE 断开即丢**；`run.json` 只存粗粒度 `outcome`，出错仅落一行 `"{Type}: {msg}"`；其余进 langfuse（排障需调其 API）。本地没有任何一个对象能回答"哪个通道、哪类错、退避几次、落在哪态"，Agent 自诊与跨 run 对比无从谈起。

## What Changes

- 引入**结果分类**（正交三轴，非扁平标签）：
  - `level`：`tool` / `model` / `task`；
  - `result`：`ok` / `degraded` / `failed`；
  - `reason`（仅 `degraded`/`failed` 有意义）：`retryable-infra`、`permanent-config`、`logic`、`control`。
- 用同一份分类统一所有 tool/model 失败路径与任务层 rollup：退避策略绑定 `reason` 而非 `level`（本 change 不实现退避策略，只确立契约，供「检索工具失败处理中间层」消费）。
- 任务层状态机收敛为 `completed / incomplete / failed / cancelled`；`degraded` 作为 **completed 的修饰**引入（含降级层级标签），不新开并列终态。
- 新增按 run 分片、持久化、类型化的本地事件轨 `logs/<run_id>.jsonl`（补齐内存事件轨不落盘缺口），供 Agent 直接 `Read` 自诊；与既有 `metrics.json` 同源喂给、避免"报表降级、trace 无事件"错位。
- 冒烟评估锚定单一现实场景（坦克毁伤评估标准）：按"问法详细程度"做 3 种变体（简单一句话 / 期望章节 / 面向 MLLM 评估），不设通用题回归哨兵；补"路由监护无效耗费"指标（期望 fast 被导到 supervisor 时记录无谓 token / 工具调用）。

## Capabilities

### New Capabilities

- `research/run-observability`: run 结果分类（level×result×reason）、任务层终态与 degraded 修饰、本地类型化事件轨、指标聚合、路由监护耗损度量。

### Modified Capabilities

无（当前 `openspec/specs/` 为空，为新建能力）。

## Impact

- 代码：`src/api/manager.py`（固化终态、持久化事件轨）、`src/api/models.py`（状态机/字段）、`src/research/eval_metrics.py`（增加 degraded×层级与 reason 分布）、`src/research/agent_store.py`（run.jsonl 落盘 / 版本字段）、新增结果分类器模块、故障注入与冒烟评测脚本。
- 外部契约：`RunStatus` 增加/调整终态语义（`degraded` 作为 completed 修饰）。
- 非目标：**不实现**退避重试策略、降级降级策略本尊、v3 全量质量评测集、图片/表格/公式索引——这些延后（见 `docs/intend.md` P0 后续 / P1 / P2）。