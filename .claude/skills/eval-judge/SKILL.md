---
name: eval-judge
description: LLM-as-judge 内容质量评估——对 eval/runtime/batches/ 某批运行结果按多维 rubric 评级（A/B/C/D + 理由 + 优缺点）。触发词：评估批次、评判输出质量、打分 run、judge 这个 batch。
---

# eval-judge：LLM-as-judge 评估运行批次

你是判官。目标批次在 `eval/runtime/batches/b_*/`，每题有 `questions.yaml` 中的人工
`expected` 判据。`scripts/eval_runtime.py` 已给出结构/遥测指标（路由、成本、引用覆盖、
交付覆盖），**不评内容质量**——这一步由你补上：对每题的**最终答案**逐维评级，重点是
**理由与优缺点**，字母等级只是附注。

判官模型 = 你自己（调用方配置的模型）。**不要**重跑 Agent，只读已落盘的运行结果。

> 维度/等级/类/建议的完整含义见 `eval/runtime/judge-rubric.md`（本 skill 的权威判据）。

## Step 0 — 定位批次与模板

```bash
python scripts/eval_judge.py --batch <id>     # 缺省取最新批次；把每题打包成 judge/case_qN.json + 报告模板
```

- 读 `eval/runtime/questions.yaml`（`question/expected/category/expected_route`）与
  `eval/runtime/batches/<id>/summary.md`（遥测总览）。
- 若 `judge/case_qN.json` 显示 `answer_truncated:true` 或 `evidence_omitted` 很大、
  嫌证据不够，加 `--token-budget 10000` 重打后再判。
- 缺 `runs/qN.json`（如 q9 只有 checkpoint）的题会自动 `skipped`，直接跳过。

## Step 1 — 选择性直读被判内容（关键）

判官输入 = 每题 `judge/case_qN.json`（已替你截好）。若想亲自核对原始字段，按此
配方**选择性**读 `runs/qN.json`（supervisor 题 1.5–1.9 MB，**绝不能整读**）：

1. `request`、`answer.content`（完整 markdown）、`answer.evidence_ids`、`answer.limitations`；
2. 只取被引用证据 `answer.evidence_ids ∩ evidence[]`，保引用顺序，每条取 `quote`+
   `source_file`/`page`/`section_path`，按 `(source_file,page)` 去重；
3. 每条引文截到 ~120–150 字符，总引文预算 ~3–4k token，超出的记 `omitted`；
4. `consistency_issues`（含 `recommendation`）；supervisor 另读 `worker_packets[].status`
   计数与 `document_plan` 章节标题。

用 `python -c` 一次性抽取，不要整读大文件。

## Step 2 — 逐维评级

对每题按维度打 **A/B/C/D**（优/良/中/差），**每个字母必须配一条具体理由 + 优点/缺点
（引用原文/证据佐证）**。理由与优缺点才是重点。

| 维度 | 判什么 | direct | borrow | unrelated |
|---|---|---|---|---|
| **正确性** | 事实是否对；direct=与源符，borrow=推断合理，unrelated=通用知识正确 | 高 | 中 | 中 |
| **忠实度/无幻觉** | 每条实质主张是否被引用证据引文支撑；有无虚构/编造引用、有无超出证据的断言 | 高 | 高 | 高 |
| **完整性** | 是否覆盖问题全部所指（supervisor=是否覆盖请求的交付面） | 中 | 高 | 低 |
| **一致性** | 答案内部/跨章节有无自相矛盾（综合 Agent 自报 consistency_issues） | 中 | 中 | 低 |
| **诚实性** | 无关题诚实声明无资料而不虚构；borrow 标注 hypothesis/设计值，不夸大确定性 | 低 | 高 | **最高** |

supervisor「标准/方案」交付额外加一维：
| **可操作性** | 规则能否落地：阈值已定义、无未定义组合、覆盖目标场景 | 仅 supervisor |

语义要点：
- **忠实度**是核心——抓「无关题强套引用」「主张无证据支撑」「编造引用」。
- **诚实性**对 unrelated 是生死线：公交调度题引 80 条军事库证据 = 失败，必须判 C/D。
- 综合 `consistency_issues`：Agent 自报 5 条（3 error）说明交付物确实不一致，别忽略。

### 输出 schema（每题）

```jsonc
{
  "question_id": "q6",
  "dimensions": [
    { "name": "正确性", "tier": "A|B|C|D", "reason": "…", "pros": ["…"], "cons": ["…"] }
  ],
  "overall_strengths": ["…"],
  "overall_weaknesses": ["…"],
  "conclusion": "…",
  "recommendation": "approve|revise|reject"
}
```

## Step 3 — 落盘报告

把各题 verdict 写进 `eval/runtime/batches/<id>/judge/report.md`：

- 逐题 section：每题 `### qN` 下用一张**表格**列逐维 `等级 | 理由 | 优点 | 缺点`（优点/缺点多项用 `<br>` 分隔），随后是总体优缺点、结论与建议；
- `## 汇总` 表：每行 `题/类/路由匹配/各维等级/建议`，表尾补等级分布与跨题模式（如
  "无关题普遍忠实度差""supervisor 一致性问题集中"）；
- 可选回写 `judge/report.json` 的 `cases[].verdict`（之后可用 `python scripts/eval_judge.py
  --report` 重新渲染 report.md）。

## 常见问题

- **缺 runs 记录**：`skipped`，不假设 1:1。
- **嫌 token 不够**：`--token-budget 10000` 重打。
- **无头自动评分**（CI）：`python scripts/eval_judge.py --api <model> --api-key <key>`，
  由脚本直接调 LLM 填报告；交互式流程不要用。