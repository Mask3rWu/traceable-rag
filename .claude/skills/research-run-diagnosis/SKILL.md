---
name: research-run-diagnosis
description: 排查研究 Agent 某次 run 的故障或质量。触发词：run 失败 / 诊断这一次运行 / 这个 run 出了什么问题 / 看 trace / 看 metrics / 为什么 failed / degraded 了。产出：定位失败的通道与 reason、判断是基础设施问题还是内容质量问题、给出该不该补发或修配置的倾向。只读排查，不改生产逻辑。
---

# research-run-diagnosis：研究 Agent 单次 run 故障排查

你是一个 run 的"读谱"者：给定一个 `run_id`，只读地把它这次的**结果分类、故障原因、降级情况**讲清楚。不改任何生产逻辑、不重跑 agent、不调外部 trace 服务（langfuse 是运维看板，本地文件才是你该先读的）。

## 一、先定位产物目录

`processed/research/agent-runs/<run_id>/`（Windows：`D:\Project\cc\processed\research\agent-runs\<run_id>\`）。每个 run 最多四个文件：

| 文件 | 何时存在 | 记什么 |
|---|---|---|
| `trace.jsonl` | **总是有**（每个事件即写，SSE 断开也不丢） | 逐行一个事件 `type/data`；失败/终态事件带 `level/result/reason` |
| `run.json` | run 完成后（agent 返回才 save） | `AgentRun` 全量：`outcome`、`route`、`evidence`、`worker_packets`、`answer`、`consistency_issues` |
| `metrics.json` | **completed / routed_away** 时 | 遥测：`model_calls`/`tool_calls`/`schema` + 新增 `outcomes`/`reason_counts`/`degraded_layers` |
| `checkpoint.json` | 续跑/章节波次后 | 续跑快照 `RunCheckpoint` |

## 二、读谱顺序（固定）

1. **先 `trace.jsonl` 末尾** —— 它是失败 run 的地基（`run.json`/`metrics.json` 在崩溃时**都不落盘**，trace 一定有）。
   找终态事件：`completed` / `failed` / `cancelled` / `routed_away`，读它的 `data.level / result / reason`。
2. 再看 `metrics.json` 的 `reason_counts`/`degraded_layers`/`outcomes`：判断是**单次偶发**还是**这波系统性**。
3. 最后（仅 completed 有）`run.json`：`outcome`、`worker_packets[*].status`、`evidence` 数、`answer.evidence_ids` 是否为空（空=引用缺失，属内容质量）。

## 三、reason 语义（判定该补发还是修配置的关键）

| reason | 含义 | 倾向 |
|---|---|---|
| `retryable-infra` | DB 断、embed/model 5xx、超时 | 基础设施抖动 → 该重试/降级，通常是临时的 |
| `permanent-config` | auth 401、配置错、模型下线 | 修配置，重发无意义 |
| `logic` | 参数校验、未知 evidence_id、schema 违反、**空证据** | 内容/契约质量问题，交 LLM 或改题，不是重试 |
| `control` | cancel、budget_reached | 正常终态，**不是故障**，别当失败处理 |

`retryable-infra` 来自 tool 还是 model 的 API 失败**是同一个 reason**——退避契约绑 reason 不绑组件。降级（dense→BM25）会标 `result=degraded`，层级在 `degraded_layers`（`retrieval/model/task`）。

## 四、已知坑（排障时会踩的）

- **崩溃 run 只有 `trace.jsonl`**：`run.json`/`metrics.json` 缺失是正常的，别据此判"数据坏了"。
- **`degraded=null` / `degraded_layers=[]` 是正常**：降级策略本尊还没实现，只有"某子组件确实降级"时才非空。
- **能用 `Read` 直读**:`trace.jsonl` 每行是 JSON,直接读尾部即可，不要先调 langfuse API。
- 中文问题体经 `curl` 会转码：提交千万别在 shell 里内联中文 JSON，写 UTF-8 文件再用 `--data-binary @file`。

## 五、验证手段（只读）

- 故障注入（不依赖 DB/模型）：`python scripts/fault_inject_outcomes.py`，覆盖 infra→failed、空证据→logic、degraded→completed+retrieval、control 不误标。
- 单元：`python -m pytest tests/test_outcome.py tests/test_api.py tests/test_eval_metrics.py -q`。
- 想要一次真实 run 的全套产物：起 `scripts/run_api.py` → 提交一个冒烟题（`eval/runtime/questions.yaml` 的坦克 q6/q11/q12）→ 完成后看该 run 目录四件套。

## 六、输出的形状

对用户给出：`run_id` → 终态 → `level×result×reason` → 单次 or 系统性 → **倾向**（该重发/该修配置/该交给 LLM 或改标注/是控制信号不用管）。如果 trace 只有部分事件就断了，明确说"run 中途崩"，别臆造剩余章节。
