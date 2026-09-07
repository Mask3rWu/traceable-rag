# Design: 结果分类与观测落地

## Context

现状(见 proposal.md - Why):失败在 `api/manager.py` 被裸 `except` 统一落成 `failed`;LangChain 事件轨(`events.py` → `manager.py._emit` → `managed.events`)纯内存、不落盘;`run.json` 只有粗 `outcome`(`agent_models.py`),错误仅一行字符串;`metrics.json`(`eval_metrics.RuntimeMetrics` → `agent_store.save_metrics`)已有 cost/phase/tool-call/schema 聚合,但没有 degraded 与 reason 分布。项目约定"只读分析/诊断脚本不改生产逻辑"。本 change 只做**分类 + 观测 + 冒烟**,不做退避策略(那是下一个 P0)。

本 change 依据 spec(`specs/research/run-observability/spec.md`)实现其行为契约。

## Goals / Non-Goals

**Goals:**
- 让每个 tool/model 失败都能归一成 `level × result × reason`,并能在本地直接读到。
- 任务层终态收敛为 4 态 + `degraded` 作为 completed 修饰(带层级标签),不新开并列终态。
- 持久化类型化事件轨,补齐"SSE 断开即丢"缺口;`metrics.json` 与事件轨同源,不漂移。
- 冒烟聚焦现实场景,补路由监护耗损指标。

**Non-Goals:**
- 不实现退避/重试、降级策略本尊(下个 P0)。
- 不建 v3 全量质量评测集,不受图片/表格/公式索引重做影响(延后至 P1/P2)。
- 不改 langfuse 的角色(继续承载运维看板),只补其缺失的本地留用通道。

## Decisions

**D1 — 结果分类做成独立纯函数模块 `src/research/outcome.py`**
输入:异常对象或软信号(`cancel_event`/`budget_reached` 标记)。输出:规范化 `(level, result, reason)`。纯函数、无 IO,便于单测驱动,也让 `manager.py` 与 `metrics` 消费同一根判断。替代:把分类内联进 manager——否决,内联会让"哪个通道哪类错"的判断逻辑与落盘/终结耦死,单测与复用都难。

**D2 — 任务层 `degraded` 用 `status: completed` + 结构化 meta 表示,不增并列终态**
API `RunStatus` 保持单调;新增可选字段如 `degraded: {layers: ["retrieval", ...]} | null` 附在 completed 上(允许多层级;单层也走数组,免得后续加第二层再改契约)。替代 a) 加字面量 `degraded_retrieval` 等——否决,违反"修饰非并列"、状态数爆炸;替代 b) 仅布尔——否决,丢失"在哪个层级降级"。

**D3 — 事件轨落盘走与现有 run 目录同根的 `trace.jsonl`,不复用内存事件流改写**
在 `PROCESSED_ROOT/research/agent-runs/<run_id>/` 下与 `run.json`/`metrics.json` 并列新增 `trace.jsonl`(类型化逐事件追加、权限/空间同现有 store)。`manager.py._emit` 在写内存 `managed.events` 的同时追加到该文件——保留现有 SSE 语义,只增补持久化。失败事件在 trace 中带 `level/result/reason`。替代:独立 `logs/` 根——否决,与既有 store 目录分裂、清理由/生命周期难对齐。

**D4 — `metrics.json` 与事件轨用同一分类函数喂,** 不各自维护一份 taxonomy
`RuntimeMetrics` 增加 `result×reason` 与 `degraded×layer` 的计数;其 `record_tool_call`/`record_model_call`/新增 `record_outcome` 内部都调用 D1 的 `classify`。这样 trace 里的 reason 与 metrics 直方图来自同一行判断点,"报表降级、trace 无事件"的错位在源头杜绝。替代:两处各写映射——否决,正是要消除的漂移源。

**D5 — 路由监护耗损指标挂在既有 `RoutePolicyError` 上**
`RoutePolicyError` 已带 `mode/reason`;新增 `wasted_tokens`、`spurious_tool_calls`,在 `manager.py` 捕获 `routed_away` 时从尚未清零的 metrics 快照取数,写入该 run 的记录。替代:单独造一条信号链——否决,守卫中断本就是 metrics 截止点。

**D6 — 故障注入用"分类正确性"断言,不模拟退避**
本 change 不做退避,注入点择在**分类边界**:让 store/retriever/model 抛 `OperationalError`/`ValueError`/触发 `cancel`,断言 trace 与 metrics 落到预期的 `level×result×reason` 且 `logic`/`control` 不被误标 `retryable-infra`。冒烟集锚定坦克毁伤评估标准单一场景,按问法详细程度做 3 变体(见 proposal.md - What Changes)。注:避免为测"重试行为"注入到整工具函数之上——那属于下个 P0 的重试边界,本 change 不越界。

## Risks / Trade-offs

- **事件轨随 run 数线性增长** → 复用现有 `agent-runs/` 分目录与计费/清理节奏,不新增全局堆积点。
- **`degraded` 挂在 completed 上可能让消费方误以为状态不扁平** → 契约文档化,客户端按 `status + degraded` 组合读取;状态机本身仍 4 态单调。
- **trace 与 metrics 双物仍可能漂移** → 由 D4 单一分类函数 + 同一 record 调用点兜底,trace 与 metrics 是同一次记录的两种投影。
- **注入点挑错会把逻辑错误当基础设施** → 注入只作用于可重试 IO 层之上/之下的分类判读,并配 `logic`/`control` 反向断言监控误标。

## Migration Plan

增量、无破坏:
1. 新增 `src/research/outcome.py`(纯函数,无对外副作用)。
2. `manager.py`/`models.py` 增补:完成路径归一化到 D2 语义、`_emit` 追加 `trace.jsonl`、捕获 `routed_away` 时写耗损。
3. `eval_metrics.py` 增加 reason/degraded 计数,由 `classify` 喂给。
4. 既有 `run.json`/`metrics.json` 字段均为**新增键**,旧记录不回填、读取兼容(缺失即不显示 degraded)。
5. 冒烟 + 故障注入脚本进 `scripts/`(只读诊断不改生产逻辑,遵守项目约定)。
回滚:删除新增键与 trace 文件即可,不影响存量运行。

## Open Questions

无——规格、方案与任务划分在本 change 范围内已自洽;其余未知项(退避策略、v3 题集、多模态索引)已明确划出 Non-Goals。