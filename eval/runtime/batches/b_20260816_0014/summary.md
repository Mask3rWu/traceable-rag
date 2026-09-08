# 运行效果评估批次 b_20260816_0014

- schema_version: `runtime-eval-v2`
- git_commit: `c7621ac5bff4b96d41acceea4508118c13f74e30`
- created_at: 2026-08-15T16:57:21Z
- 问题数: 10

## 逐题结果

| 题 | 类 | 路由匹配 | outcome | 耗时(s) | 模型次数 | 成本(¥) | 工具次数 | 检索次数 | 去重引用 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| #1 q1 | direct | fast/fast ✓ | completed | 45.2 | 4 | 0.0238 | 4 | 2 | 5 |
| #2 q2 | direct | fast/fast ✓ | completed | 45.21 | 6 | 0.0685 | 7 | 6 | 15 |
| #3 q3 | borrow | fast/fast ✓ | incomplete | 45.2 | 9 | 0.0813 | 15 | 14 | 0 |
| #4 q4 | borrow | fast/supervisor ✗ | routed_away | 9.03 | 0 | — | 0 | 0 | 0 |
| #5 q5 | unrelated | fast/fast ✓ | completed | 45.24 | 8 | 0.0974 | 12 | 10 | 1 |
| #6 q6 | direct | supervisor/— — | failed | 1143.35 | 0 | — | 0 | 0 | 0 |
| #7 q7 | direct | supervisor/supervisor ✓ | completed | 1780.24 | 126 | 3.1123 | 245 | 184 | 78 |
| #8 q8 | borrow | supervisor/supervisor ✓ | completed | 2055.0 | 125 | 3.4923 | 235 | 162 | 66 |
| #9 q9 | borrow | supervisor/supervisor ✓ | completed | 2521.0 | 148 | 4.6902 | 272 | 210 | 130 |
| #10 q10 | unrelated | supervisor/— — | failed | 696.75 | 0 | — | 0 | 0 | 0 |

## 汇总

- 状态分布: completed 6, incomplete 1, failed 2, routed_away 1, cancelled 0
- 模型调用: 426 次，成本 ¥11.5658；schema 校验 58 次，通过 52，失败 6
- 工具调用: 790 次，成功率 98.7%
- 检索: 588 次，去重前 5378，去重后 1758，实际引用 295
- 交付覆盖: 规划章节 23，执行 packet 23（sufficient 23 / insufficient 0 / failed 0 / blocked 0），已组装 7/10 题

## 检索质量（深度 8，弱金=被采纳证据）

- 被采纳证据 268：Dense 排前 48.5%（130）、BM25 排前 42.5%（114）、并列 24
- 单路救援：仅 Dense 救回 21.3%（57）、仅 BM25 21.6%（58）、两路均在深度外仅融合 22.4%（60）、两路均在深度内 34.7%（93）

| 题 | 被采纳 | Dense前 | BM25前 | 仅Dense救 | 仅BM25救 | 融合only | 两路均稳 |
|---|---|---:|---:|---:|---:|---:|---:|
| #1 q1 | 5 | 1 | 4 | 1 | 3 | 0 | 1 |
| #2 q2 | 14 | 7 | 5 | 3 | 2 | 4 | 5 |
| #5 q5 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| #7 q7 | 66 | 39 | 19 | 16 | 15 | 13 | 22 |
| #8 q8 | 57 | 30 | 25 | 15 | 10 | 14 | 18 |
| #9 q9 | 125 | 53 | 60 | 22 | 28 | 29 | 46 |

## 路由守卫中断（误判时立即中断）
- #4 q4: router 决策 mode='supervisor'，reason='请求涉及核爆环境与舰船结构易损性多源资料的综合归纳，需系统整理因素并形成评估框架，属于多部分研究报告。'

## 失败明细
- #10 q10: BadRequestError: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. (insufficient tool messages following tool_calls message)", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}} （696.75s）
- #6 q6: BadRequestError: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. (insufficient tool messages following tool_calls message)", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}} （1143.35s）

## 阶段成本归因（全批次汇总）

| 阶段 | 参与题数 | 模型调用 | prompt(tok) | completion(tok) | 成本(¥) | 总耗时(s) | 失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| router | 7 | 7 | 1447 | 795 | 0.0064 | 13437.8 | 0 |
| fast | 4 | 23 | 253305 | 8484 | 0.2670 | 81307.2 | 0 |
| planner | 3 | 3 | 3103 | 22447 | 0.1360 | 170018.8 | 0 |
| worker | 3 | 206 | 6254032 | 349068 | 5.2707 | 2805457.8 | 0 |
| review | 3 | 3 | 43660 | 42326 | 0.3815 | 340995.4 | 0 |
| unknown | 3 | 184 | 5943406 | 415191 | 5.5041 | 3215166.5 | 0 |
