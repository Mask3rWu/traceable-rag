# 运行效果评估批次 b_20260811_1957_backfill

- schema_version: `runtime-eval-v2`
- git_commit: `89b1d6d3e79547a8d691fd70525352342d6a9549`
- created_at: 2026-08-11T13:59:52Z
- 问题数: 10

## 逐题结果

| 题 | 类 | 路由匹配 | outcome | 耗时(s) | 模型次数 | 成本(¥) | 工具次数 | 检索次数 | 去重引用 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| #1 q1 | direct | fast/fast ✓ | completed | 66.25 | 4 | 0.0494 | 5 | 4 | 7 |
| #2 q2 | direct | fast/fast ✓ | completed | 66.41 | 9 | 0.2570 | 15 | 13 | 15 |
| #3 q3 | borrow | fast/fast ✓ | incomplete | 66.21 | 9 | 0.3665 | 16 | 15 | 0 |
| #4 q4 | borrow | fast/supervisor ✗ | routed_away | 66.19 | 1 | 0.0014 | 0 | 0 | 0 |
| #5 q5 | unrelated | fast/fast ✓ | completed | 66.22 | 8 | 0.3299 | 12 | 10 | 1 |
| #6 q6 | direct | supervisor/supervisor ✓ | completed | 2017.17 | 116 | 15.8698 | 234 | 164 | 107 |
| #7 q7 | direct | supervisor/supervisor ✓ | completed | 1972.75 | 100 | 14.7273 | 204 | 146 | 128 |
| #8 q8 | borrow | supervisor/supervisor ✓ | completed | 2189.15 | 109 | 14.7325 | 204 | 140 | 125 |
| #9 q9 | borrow | supervisor/— — | failed | 539.68 | 6 | 0.8097 | 3 | 0 | 43 |
| #10 q10 | unrelated | supervisor/supervisor ✓ | completed | 1952.12 | 88 | 11.0854 | 173 | 111 | 80 |

## 汇总

- 状态分布: completed 7, incomplete 1, failed 1, routed_away 1, cancelled 0
- 模型调用: 450 次，成本 ¥58.2289；schema 校验 0 次，通过 0，失败 0
- 工具调用: 866 次，成功率 95.5%
- 检索: 603 次，去重前 5893，去重后 2415，实际引用 506
- 交付覆盖: 规划章节 38，执行 packet 33（sufficient 33 / insufficient 0 / failed 0 / blocked 0），已组装 8/10 题

## 路由守卫中断（误判时立即中断）
- #4 q4: router 决策 mode='supervisor'，reason='Requires synthesizing multiple environmental and structural factors into a structured damage-level assessment summary.'

## 失败明细
- #9 q9: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. (insufficient tool messages following tool_calls message)", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}} （539.68s）

## 阶段成本归因（全批次汇总）

| 阶段 | 参与题数 | 模型调用 | prompt(tok) | completion(tok) | 成本(¥) | 总耗时(s) | 失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| router | 4 | 4 | 825 | 602 | 0.0061 | 7778.9 | 0 |
| fast | 4 | 26 | 311753 | 10231 | 0.9966 | 97772.1 | 0 |

## 补填说明（一次性离线回填）

- 来源批次: `b_20260811_1957`（schema runtime-eval-v1），本次升级为 `runtime-eval-v2`。
- 保留原 op 记录：model_calls / tool_calls / retrieval / elapsed_s；`q4`(routed_away)、`q9`(failed) 因当时 metrics 未落盘，计数由 Langfuse trace 重建，属下限估计。
- `delivery`(交付覆盖) / `route_matched`(路由匹配) / `steps_retries` / `phase_cost` / `error` / `ended_at` 为本批新增字段。
- `phase_cost` 仅覆盖 fast 段（router + fast 两步）：取本地 metrics.json，首个模型调用=router、其余=fast；supervisor 段无 phase tag 且历史 trace 的 worker 阶段无法从 span 名归因，记 null。
- `plan_retries` 历史未记录 schema 校验，不可恢复，记 null。
- `error` 取自 Langfuse ERROR span 的 statusMessage；`ended_at` 取自 trace 最后 observation 的 endTime。

### 成本单位与近似来源说明

- 全部成本已换算为人民币，单价按 deepseek-v4-flash：`input ¥3 / 百万 token`、`output ¥6 / 百万 token`（`input_cache_hit ¥0.025` 因历史无缓存字段未参与计费）。
- **历史运行未记录 prompt 缓存命中 token，全部 prompt 按 cache miss 单价（¥3/M）计，未计入前缀缓存折扣**，故本批人民币成本为上限估计；实际运行若命中缓存会低于此值。
- `phase_cost` 的 router/fast 归因基于“fast 图首个模型调用即 router、其余即 fast”的结构假设（fast 图结构确定，视为近似精确）。
- `q4`/`q9` 的 model_calls / tool_calls / 检索计数及成本由 Langfuse trace 重建（当时 metrics 未落盘）；trace 对部分运行只捕获前半段 generation，此类数值属近似/下限估计。
- 单价本身按当前配置的 deepseek-v4-flash 人民币计费填写，若实际账单有出入以账单为准。
