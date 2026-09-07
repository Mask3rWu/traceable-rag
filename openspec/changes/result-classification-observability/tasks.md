# Tasks: 结果分类与观测落地

## 1. 结果分类器

- [x] 1.1 新建 `src/research/outcome.py`，实现纯函数 `classify(exc | soft_signal) -> (level, result, reason)`，覆盖 `retryable-infra / permanent-config / logic / control` 四类 reason 与 `ok/degraded/failed` 三种 result；用 `pytest` 对异常类型→三元组做参数化单测，断言 `OperationalError`/embed 5xx→`(..., failed, retryable-infra)`、`ValueError`(未知 evidence_id)→`(…, logic)`、cancel/budget→`control` 且不被标 retryable-infra。
- [x] 1.2 暴露分类入口给 manager 与 metrics 共用（`outcome.classify` 单一调用点），并加文档字符串说明输入域（异常 vs 软信号）——验证：该模块除纯函数外无 IO，`pytest tests/` 关于 outcome 的全绿。

## 2. 状态机与 degraded 语义

- [x] 2.1 在 `src/api/models.py` 的 `RunStatus`/`RunSummary` 增加可选 `degraded: {layers: [...]} | null`，语义上仅允许随 `completed` 出现；保持 4 个并列终态 `completed/incomplete/failed/cancelled` 不变——验证：pydantic 用例断言 `status=completed` 可带 `degraded.layers=["retrieval"]`，而 `degraded` 与 `failed` 组合被拒绝。
- [x] 2.2 在 `api/manager.py` 完成路径把"任一子组件返回 degraded 但任务完整"归一化为 `status=completed` + `degraded.layers`（并入 `_execute` 的结果分支），并保留现有 cancel 优先语义——验证：单测注入一个 worker 降级、其余 completed，断言最终 `status=completed`、`degraded.layers` 含检索层级；`failed` 分支仍非 degraded。

## 3. 事件轨落盘

- [x] 3.1 在既有 run 目录（`PROCESSED_ROOT/research/agent-runs/<run_id>/`）新增 `trace.jsonl` 持久化：让 `manager.py._emit` 在写内存 `managed.events` 的同时追加同一 `RunEvent` 到该文件（类型化、追加式、含 sequence/type/data）——验证：跑一次后 `trace.jsonl` 存在且逐行可 `json.loads`，行数>=内存事件数。
- [x] 3.2 失败类事件在 trace 中带 `level/result/reason` 字段（由 1 的分类喂给），并在 run 终态落一条定型事件——验证：故障注入一个失败 run 后，`trace.jsonl` 末尾事件含上述三元组，可 `Read` 直接定位通道与 reason。

## 4. 指标与事件同源

- [x] 4.1 扩展 `src/research/eval_metrics.py` 的 `RuntimeMetrics`：新增 `record_outcome`（内部调用 `outcome.classify`），聚合 `degraded×layer` 计数与 `reason` 分布；`tool_calls_summary`/`model_calls_summary` 已含的计数保留——验证：单测对同一 failure 同时喂给事件轨与 metrics，断言两者 reason 一致（同源不漂移）。
- [x] 4.2 在 `agent_store.save_metrics` 落盘中体现新增键（缺失兼容不报错）——验证：旧记录加载不回填、新记录含 `result_reason` 直方图字段。

## 5. 路由监护耗损

- [x] 5.1 给 `RoutePolicyError` 增加 `wasted_tokens`、`spurious_tool_calls` 字段，并在 `manager.py` 捕获 `routed_away` 时从 metrics 快照取数写入该 run 记录——验证：构造"期望 fast 却被导到 supervisor"的 run，断言 run 记录里 `wasted_tokens>0` 且工具调用计数=误判前实际发生次数。

## 6. 故障注入与冒烟

- [x] 6.1 新建 `scripts/` 冒烟/故障注入脚本（只读诊断、不改生产逻辑）：注入 store/retriever/model 的 `OperationalError`/`ValueError`/cancel，断言 trace 与 metrics 落到预期 `level×result×reason`，含 `logic`/`control` 不误标 `retryable-infra` 的反向断言——验证：脚本对每类注入输出 PASS，且全量后 exit 0。
- [x] 6.2 冒烟题集锚定单一现实场景——坦克毁伤评估标准：在 `eval/runtime/questions.yaml` 处为该场景配置 3 种"问法详细程度"变体——简单版（如"输出一份坦克的毁伤评估标准"）、详细版（如当前 q6，给出期望章节）、面向 MLLM 评估版（要求产出可供 MLLM 直接评估用的坦克毁伤评估标准）；不设通用题回归哨兵——验证：跑通 `scripts/eval_runtime.py` 时三变体题各自得到 `completed` 终态，且可对着同一参考答案 diff 比较。
- [x] 6.3 冒烟集能区分"内容质量问题"与"基础设施失败"：同为 `failed`，`reason` 不同（infra vs logic）——验证：故障注入一个 DB 断、另造一个模型空证据，两 run 的 trace `reason` 分别为 `retryable-infra/permanent-config` 与 `logic`，断言结果区分。

## 7. 契约与文档收尾

- [x] 7.1 更新 `docs/intend.md` 的 P0「结果分类与观测落地」小节，标注已完成项并记录落地后的实际指标/故障演练结果（按"生产化成果的简历边界"口径，不夸大）——验证：文档中 P0 方案条目对已落地项补标完成、未落地项仍标计划。
- [x] 7.2 代码 docstring 引用设计文档只写章节语义、不写文件名（遵守项目约定）——验证：grep 新增注释中无 "design.md/文件名" 字样。