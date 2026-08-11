# 运行效果评估批次 b_20260811_1957

- schema_version: `runtime-eval-v1`
- git_commit: `ccb9e0cb5a5fd62656e28ac3723895d3014ea0d8`
- created_at: 2026-08-11T12:35:27Z
- 问题数: 10

## 逐题结果

| 题 | 类 | outcome | 耗时(s) | 模型次数 | 成本($) | 工具次数 | 检索次数 | 去重引用 |
|---|---|---|---|---:|---:|---:|---:|---:|
| #1 q1 | direct | completed | 66.25 | 4 | 0.0054 | 5 | 4 | 7 |
| #2 q2 | direct | completed | 66.41 | 9 | 0.0253 | 15 | 13 | 15 |
| #3 q3 | borrow | incomplete | 66.21 | 9 | 0.0347 | 16 | 15 | 0 |
| #4 q4 | borrow | routed_away | 66.19 | 0 | — | 0 | 0 | 0 |
| #5 q5 | unrelated | completed | 66.22 | 8 | 0.0308 | 12 | 10 | 1 |
| #6 q6 | direct | completed | 2017.17 | 116 | 1.6095 | 234 | 164 | 107 |
| #7 q7 | direct | completed | 1972.75 | 100 | 1.4900 | 204 | 146 | 128 |
| #8 q8 | borrow | completed | 2189.15 | 109 | 1.5005 | 204 | 140 | 125 |
| #9 q9 | borrow | failed | 539.68 | 0 | — | 0 | 0 | 0 |
| #10 q10 | unrelated | completed | 1952.12 | 88 | 1.1412 | 173 | 111 | 80 |

## 汇总

- 完成: 9 / 10，失败: 1
- 模型调用: 443 次，成本 $5.8376；schema 校验 0 次，通过 0，失败 0
- 工具调用: 863 次，成功率 95.5%
- 检索: 603 次，去重前 5622，去重后 2250，实际引用 463

## 路由守卫中断（fast 被误判为多 agent）
- #4 q4: router 决策 mode='supervisor'，reason='Requires synthesizing multiple environmental and structural factors into a structured damage-level assessment summary.'
