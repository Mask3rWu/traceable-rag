# 运行效果评估批次 b_20260817_0028

- schema_version: `runtime-eval-v2`
- git_commit: `85fc34b6195ffb0b2dcc072e390be1976849a34b`
- created_at: 2026-08-16T16:48:54Z
- 问题数: 10

## 逐题结果

| 题 | 类 | 路由匹配 | outcome | 耗时(s) | 模型次数 | 成本(¥) | 工具次数 | 检索次数 | 去重引用 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| #1 q1 | direct | fast/fast ✓ | completed | 33.17 | 4 | 0.0259 | 4 | 2 | 5 |
| #2 q2 | direct | fast/fast ✓ | completed | 33.12 | 5 | 0.0306 | 6 | 4 | 8 |
| #3 q3 | borrow | fast/fast ✓ | incomplete | 45.19 | 9 | 0.0912 | 16 | 15 | 0 |
| #4 q4 | borrow | fast/supervisor ✗ | routed_away | 33.2 | 0 | — | 0 | 0 | 0 |
| #5 q5 | unrelated | fast/fast ✓ | completed | 33.18 | 8 | 0.0836 | 11 | 9 | 4 |
| #6 q6 | direct | supervisor/— — | failed | 696.71 | 0 | — | 0 | 0 | 0 |
| #7 q7 | direct | supervisor/supervisor ✓ | completed | 802.71 | 35 | 1.0339 | 57 | 35 | 70 |
| #8 q8 | borrow | supervisor/— — | failed | 1197.78 | 0 | — | 0 | 0 | 0 |
| #9 q9 | borrow | supervisor/— — | failed | 473.36 | 0 | — | 0 | 0 | 0 |
| #10 q10 | unrelated | supervisor/supervisor ✓ | completed | 980.38 | 43 | 1.3801 | 68 | 44 | 61 |

## 汇总

- 状态分布: completed 5, incomplete 1, failed 3, routed_away 1, cancelled 0
- 模型调用: 104 次，成本 ¥2.6453；schema 校验 21 次，通过 20，失败 1
- 工具调用: 162 次，成功率 99.4%
- 检索: 109 次，去重前 1043，去重后 609，实际引用 148
- 交付覆盖: 规划章节 15，执行 packet 15（sufficient 15 / insufficient 0 / failed 0 / blocked 0），已组装 6/10 题

## 路由守卫中断（误判时立即中断）
- #4 q4: router 决策 mode='supervisor'，reason='请求要求归纳核爆环境与舰船结构易损性等多方面因素，属于需要综合分析的结构化任务。'

## 失败明细
- #9 q9: BadRequestError: Error code: 400 - {'error': {'message': "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. (insufficient tool messages following tool_calls message)", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}} （473.36s）
- #6 q6: OutputParserException: Invalid json output: 
For troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE  （696.71s）
- #8 q8: LengthFinishReasonError: Could not parse response content as the length limit was reached - CompletionUsage(completion_tokens=65534, prompt_tokens=15325, total_tokens=80859, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=62047, rejected_prediction_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cache_write_tokens=None, cached_tokens=1664), prompt_cache_hit_tokens=1664, prompt_cache_miss_tokens=13661) （1197.78s）

## 阶段成本归因（全批次汇总）

| 阶段 | 参与题数 | 模型调用 | prompt(tok) | completion(tok) | 成本(¥) | 总耗时(s) | 失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| router | 6 | 6 | 1249 | 749 | 0.0060 | 10852.3 | 0 |
| fast | 4 | 22 | 237390 | 7174 | 0.2270 | 70871.2 | 0 |
| planner | 2 | 2 | 2078 | 16334 | 0.0981 | 125682.0 | 0 |
| worker | 2 | 72 | 1209373 | 149104 | 1.8236 | 1216654.6 | 0 |
| review | 2 | 2 | 29244 | 67952 | 0.4905 | 482476.1 | 0 |
