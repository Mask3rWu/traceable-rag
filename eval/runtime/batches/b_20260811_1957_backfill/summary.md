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

## 检索质量（深度 8，弱金=被采纳证据）

- 被采纳证据 444：Dense 排前 44.4%（197）、BM25 排前 47.3%（210）、并列 37
- 单路救援：仅 Dense 救回 23.4%（104）、仅 BM25 21.2%（94）、两路均在深度外仅融合 25.9%（115）、两路均在深度内 29.5%（131）

| 题 | 被采纳 | Dense前 | BM25前 | 仅Dense救 | 仅BM25救 | 融合only | 两路均稳 |
|---|---|---:|---:|---:|---:|---:|---:|
| #1 q1 | 7 | 5 | 2 | 2 | 1 | 0 | 4 |
| #2 q2 | 15 | 7 | 6 | 5 | 1 | 4 | 5 |
| #5 q5 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| #6 q6 | 86 | 41 | 41 | 24 | 19 | 23 | 20 |
| #7 q7 | 114 | 54 | 53 | 27 | 21 | 31 | 35 |
| #8 q8 | 112 | 42 | 58 | 18 | 30 | 32 | 32 |
| #9 q9 | 43 | 21 | 19 | 13 | 11 | 8 | 11 |
| #10 q10 | 66 | 27 | 31 | 15 | 11 | 17 | 23 |

## 路由守卫中断（误判时立即中断）
- #4 q4: router 决策 mode='supervisor'，reason='Requires synthesizing multiple environmental and structural factors into a structured damage-level assessment summary.'

## 失败明细
- #9 q9: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. (insufficient tool messages following tool_calls message)", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}} （539.68s）

## 阶段成本归因（全批次汇总）

| 阶段 | 参与题数 | 模型调用 | prompt(tok) | completion(tok) | 成本(¥) | 总耗时(s) | 失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| router | 4 | 4 | 825 | 602 | 0.0061 | 7778.9 | 0 |
| fast | 4 | 26 | 311753 | 10231 | 0.9966 | 97772.1 | 0 |
