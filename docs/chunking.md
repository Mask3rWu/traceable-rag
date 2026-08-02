# 结构化 Chunk 说明

Chunk 层读取解析层正式产物 `doc.json`，可选读取同目录下的
`visual_enrichment.json`，并独立写出 `chunks.jsonl`。它不会改写解析事实。

## 策略

- 使用完整 `section_path` 作为硬边界，不跨章节合并。
- 仅包含标题的父级节点不生成独立 chunk；其完整标题路径会写入每个有正文的后代 chunk。
- `embedding_text` 是向量化输入，包含文档名、完整标题路径及正文；`text` 只保留正文，不含 Markdown 标题标记。
- 以段落、列表、公式、表格、图和图注为语义单元。
- 标题与本章节首个内容单元、图表与图注、跨页续接块保持强绑定。
- 正文引用同章节公式时，公式与解释性正文保持强绑定。
- 默认目标 600 字、上限 800 字；单个表格、公式及强绑定单元允许超限。
- 只有超长普通文本会按句切分；相邻 chunk 默认保留 80 字轻量重叠。
- MLLM 描述写入 `visual_text`，不混入解析事实 `text`。
- 视觉增强缺失、失败或裁图缺失时继续生成文本 chunk，并标记
  `visual_unavailable`。

## 输出契约

每行是一个 JSON 对象，核心字段为：

| 字段 | 含义 |
|---|---|
| `schema_version` | Chunk 字段契约版本，当前为 `2` |
| `chunk_id` | 文档内稳定顺序 ID |
| `text` | 来自 `doc.json` 的主证据文本 |
| `embedding_text` | 用于向量化的规范化文本，含文档名、完整标题路径和正文 |
| `visual_text` | 可选、低权重的 MLLM 检索辅助文本 |
| `overlap_text` | 上一个 chunk 的轻量文本重叠 |
| `block_ids` | 主内容对应的解析块 |
| `overlap_block_ids` | 重叠文本对应的解析块 |
| `page_start` / `page_end` | 来源页范围 |
| `section_path` | 所属章节路径 |
| `heading_path` | 人类可读的完整标题路径，供检索上下文和展示使用 |
| `references` | 正文引用的块 ID |
| `visual_assets` | chunk 内或被引用图表的裁图、描述和关系 |
| `quality_flags` | 无效关系、视觉不可用等非阻塞告警 |

## 使用

处理指定文档：

```powershell
conda run -n dba-py311 python scripts/build_chunks.py processed/parsed/2-电子系统/doc.json
```

处理全部解析文档：

```powershell
conda run -n dba-py311 python scripts/build_chunks.py --all
```

可用 `--target-chars`、`--max-chars` 和 `--overlap-chars` 调整首版字符阈值。
